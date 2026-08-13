#!/usr/bin/env python3
"""
Phase 2: Sub-Reward Network (SRN) Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trains the SRN to evaluate generated sign-language videos along four
complementary axes.  The frozen diffusion model from Phase 1 generates
K=8 candidate videos per sentence; quality metrics (MPJPE, temporal
smoothness, CLIP similarity) establish ground-truth rankings; and the
SRN is trained to reproduce these rankings via listwise ranking loss.

Loss:
    L = -E[ log( exp(r_i / tau) / sum_j exp(r_j / tau) ) ]

Usage:
    python -m signrl_diff.scripts.train_phase2 \
        --config configs/default.yaml \
        --data_dir ./data \
        --diffusion_checkpoint ./checkpoints/phase1/final.pt \
        --output_dir ./checkpoints/phase2
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from signrl_diff.models.diffusion import VideoUNet, AutoencoderKL, DDPMScheduler
from signrl_diff.models.srn import SubRewardNetwork, listwise_ranking_loss
from signrl_diff.data import SignLanguageVideoDataset
from signrl_diff.utils import set_seed, count_parameters, save_checkpoint, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_phase2")


# ======================================================================
# Quality Metric Computation
# ======================================================================

def compute_mpjpe_score(
    keypoints: torch.Tensor,
    reference_keypoints: torch.Tensor,
) -> float:
    """Compute Mean Per-Joint Position Error between generated and reference.

    Lower MPJPE indicates better pose quality.

    Parameters
    ----------
    keypoints : Tensor, shape ``(T, J, 3)``
        Estimated keypoints from generated video.
    reference_keypoints : Tensor, shape ``(T, J, 3)``
        Reference keypoints (from ground-truth or average pose).

    Returns
    -------
    float
        MPJPE value (lower is better).
    """
    diff = (keypoints - reference_keypoints).pow(2).sum(dim=-1).sqrt()
    return diff.mean().item()


def compute_temporal_smoothness(keypoints: torch.Tensor) -> float:
    """Compute temporal smoothness as negative mean frame-to-frame delta.

    Higher (less negative) indicates smoother motion.

    Parameters
    ----------
    keypoints : Tensor, shape ``(T, J, 3)``
        Sequence of keypoints.

    Returns
    -------
    float
        Smoothness score (higher is better).
    """
    if keypoints.shape[0] < 2:
        return 0.0
    deltas = keypoints[1:] - keypoints[:-1]
    delta_norms = deltas.pow(2).sum(dim=-1).sqrt()
    return -delta_norms.mean().item()


def compute_clip_similarity(
    text_emb: torch.Tensor,
    video_features: torch.Tensor,
    projection: nn.Linear,
) -> float:
    """Compute CLIP-style cosine similarity between text and video.

    Parameters
    ----------
    text_emb : Tensor, shape ``(D_text,)``
        Text embedding (pooled).
    video_features : Tensor, shape ``(D_video,)``
        Video embedding (pooled).
    projection : nn.Linear
        Projects video features into text embedding space.

    Returns
    -------
    float
        Cosine similarity in [-1, 1] (higher is better).
    """
    with torch.no_grad():
        video_proj = projection(video_features.unsqueeze(0)).squeeze(0)
        text_norm = F.normalize(text_emb.unsqueeze(0), p=2, dim=-1).squeeze(0)
        video_norm = F.normalize(video_proj, p=2, dim=-1)
        similarity = (text_norm * video_norm).sum().item()
    return similarity


def estimate_keypoints_from_video(
    video: torch.Tensor,
    num_joints: int = 148,
) -> torch.Tensor:
    """Estimate pseudo-keypoints from video frames using spatial statistics.

    Parameters
    ----------
    video : Tensor, shape ``(T, 3, H, W)``
        Video tensor.
    num_joints : int
        Number of joints to estimate.

    Returns
    -------
    Tensor, shape ``(T, num_joints, 3)``
        Estimated keypoints.
    """
    T, C, H, W = video.shape
    frame_feats = video.mean(dim=1)  # (T, H, W)

    h_coords = torch.linspace(-1, 1, H, device=video.device)
    w_coords = torch.linspace(-1, 1, W, device=video.device)
    h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing="ij")

    weights = F.softmax(frame_feats.reshape(T, -1), dim=-1)
    weights_2d = weights.reshape(T, H, W)

    cx = (weights_2d * w_grid.unsqueeze(0)).sum(dim=[1, 2])  # (T,)
    cy = (weights_2d * h_grid.unsqueeze(0)).sum(dim=[1, 2])  # (T,)

    joint_offsets_x = torch.linspace(-0.5, 0.5, num_joints, device=video.device)
    joint_offsets_y = torch.linspace(-0.8, 0.8, num_joints, device=video.device)

    kpt_x = cx.unsqueeze(-1) + joint_offsets_x.unsqueeze(0) * 0.3
    kpt_y = cy.unsqueeze(-1) + joint_offsets_y.unsqueeze(0) * 0.3
    kpt_z = torch.zeros(T, num_joints, device=video.device)

    keypoints = torch.stack([kpt_x, kpt_y, kpt_z], dim=-1)
    return keypoints


# ======================================================================
# Candidate Generation
# ======================================================================

@torch.no_grad()
def generate_candidates(
    unet: nn.Module,
    vae: nn.Module,
    scheduler: nn.Module,
    text_embedding: torch.Tensor,
    K: int = 8,
    num_frames: int = 16,
    latent_channels: int = 4,
    latent_hw: int = 32,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Generate K candidate videos for a single text prompt.

    Parameters
    ----------
    unet : nn.Module
        Frozen diffusion UNet.
    vae : nn.Module
        Frozen VAE.
    scheduler : nn.Module
        DDPM scheduler.
    text_embedding : Tensor, shape ``(L, D)``
        Text embedding for the sentence.
    K : int
        Number of candidates.
    num_frames : int
        Temporal frames in latent.
    latent_channels : int
    latent_hw : int
    device : torch.device

    Returns
    -------
    list of Tensor
        K video tensors, each ``(T, 3, H, W)``.
    """
    candidates = []
    text_batch = text_embedding.unsqueeze(0).to(device)

    for k_idx in range(K):
        torch.manual_seed(k_idx * 1000 + int(time.time() * 1000) % 10000)

        z = torch.randn(
            1, num_frames, latent_channels, latent_hw, latent_hw,
            device=device,
        )

        for t in reversed(range(scheduler.num_train_steps)):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            eps_hat = unet(z, t_tensor, text_batch)
            z = scheduler.step(eps_hat, t, z)

        if hasattr(vae, "decode"):
            video = vae.decode(z)
        else:
            B, T, C, H, W = z.shape
            z_flat = z.reshape(B * T, C, H, W)
            v_flat = vae(z_flat)
            _, Co, Ho, Wo = v_flat.shape
            video = v_flat.reshape(B, T, Co, Ho, Wo)

        candidates.append(video.squeeze(0).cpu())

    return candidates


# ======================================================================
# Quality Scoring and Ranking
# ======================================================================

def compute_quality_scores(
    candidates: List[torch.Tensor],
    reference_video: torch.Tensor,
    text_embedding: torch.Tensor,
    clip_projection: nn.Linear,
    num_joints: int = 148,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute quality scores and rankings for K candidates.

    Three metrics are computed for each candidate:
    1. MPJPE (lower = better)
    2. Temporal smoothness (higher = better)
    3. CLIP similarity (higher = better)

    Scores are combined into a single quality score, and rankings are
    derived (0 = best).

    Parameters
    ----------
    candidates : list of Tensor
        K video tensors, each ``(T, 3, H, W)``.
    reference_video : Tensor, shape ``(T, 3, H, W)``
        Ground-truth video for MPJPE comparison.
    text_embedding : Tensor, shape ``(L, D)``
        Text embedding for CLIP similarity.
    clip_projection : nn.Linear
        Video-to-text projection for CLIP similarity.
    num_joints : int
    device : torch.device

    Returns
    -------
    scores : Tensor, shape ``(K,)``
        Combined quality scores (higher = better).
    rankings : Tensor, shape ``(K,)``
        Ranking indices (0 = best).
    """
    K = len(candidates)
    ref_kpts = estimate_keypoints_from_video(
        reference_video.to(device), num_joints
    )
    text_pooled = text_embedding.mean(dim=0).to(device)
    if text_pooled.shape[-1] > 512:
        text_pooled = text_pooled[:512]
    elif text_pooled.shape[-1] < 512:
        text_pooled = F.pad(text_pooled, (0, 512 - text_pooled.shape[-1]))

    mpjpe_scores = []
    smoothness_scores = []
    clip_scores = []

    for candidate in candidates:
        cand_kpts = estimate_keypoints_from_video(
            candidate.to(device), num_joints
        )

        mpjpe = compute_mpjpe_score(cand_kpts, ref_kpts)
        mpjpe_scores.append(-mpjpe)  # Negate: higher = better

        smoothness = compute_temporal_smoothness(cand_kpts)
        smoothness_scores.append(smoothness)

        cand_pooled = candidate.to(device).mean(dim=[0, 1, 2, 3])
        if cand_pooled.shape[-1] < 1024:
            cand_pooled = F.pad(cand_pooled, (0, 1024 - cand_pooled.shape[-1]))
        clip_sim = compute_clip_similarity(
            text_pooled[:512], cand_pooled[:1024], clip_projection
        )
        clip_scores.append(clip_sim)

    mpjpe_t = torch.tensor(mpjpe_scores, dtype=torch.float32)
    smooth_t = torch.tensor(smoothness_scores, dtype=torch.float32)
    clip_t = torch.tensor(clip_scores, dtype=torch.float32)

    # Normalize each metric to [0, 1]
    def normalize(x: torch.Tensor) -> torch.Tensor:
        rng = x.max() - x.min()
        if rng < 1e-8:
            return torch.zeros_like(x)
        return (x - x.min()) / rng

    combined = (
        0.4 * normalize(mpjpe_t)
        + 0.3 * normalize(smooth_t)
        + 0.3 * normalize(clip_t)
    )

    # Rankings: argsort of argsort gives rank (0 = best = highest score)
    rankings = combined.argsort(descending=True).argsort().float()

    return combined, rankings


# ======================================================================
# Main Training Loop
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: SRN Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--diffusion_checkpoint", type=str, required=True,
                        help="Path to Phase 1 checkpoint (UNet + VAE)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase2",
                        help="Directory for saving checkpoints and logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {}

    model_cfg = config.get("model", {})
    vae_cfg = config.get("vae", {})
    diff_cfg = config.get("diffusion", {})
    srn_cfg = config.get("srn", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build and load frozen diffusion models
    # ------------------------------------------------------------------
    unet = VideoUNet(
        in_channels=vae_cfg.get("latent_channels", 4),
        channel_config=model_cfg.get("unet_channels", [128, 256, 512, 512]),
        text_dim=model_cfg.get("text_dim", 1024),
        cond_dim=model_cfg.get("cond_dim", 512),
        num_heads=model_cfg.get("num_heads", 8),
    )
    vae = AutoencoderKL(
        latent_channels=vae_cfg.get("latent_channels", 4),
        base_channels=vae_cfg.get("base_channels", 64),
        kl_weight=vae_cfg.get("kl_weight", 1e-4),
    )
    scheduler = DDPMScheduler(
        num_train_steps=diff_cfg.get("num_train_steps", 1000),
        beta_schedule=diff_cfg.get("beta_schedule", "linear"),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
    )

    # Load Phase 1 checkpoint
    ckpt_path = Path(args.diffusion_checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["model_state_dict"])
        if "vae_state_dict" in ckpt:
            vae.load_state_dict(ckpt["vae_state_dict"])
        logger.info(f"Loaded diffusion checkpoint from {ckpt_path}")
    else:
        logger.warning(f"Diffusion checkpoint not found: {ckpt_path}, using random init")

    # Freeze UNet and VAE
    for param in unet.parameters():
        param.requires_grad = False
    unet.eval()

    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()

    unet = unet.to(device)
    vae = vae.to(device)
    scheduler = scheduler.to(device)

    # ------------------------------------------------------------------
    # Initialize trainable SRN
    # ------------------------------------------------------------------
    srn = SubRewardNetwork(
        num_joints=srn_cfg.get("num_joints", 148),
        video_emb_dim=srn_cfg.get("video_emb_dim", 1024),
        use_video_extractor=True,
        use_hand_extractor=True,
    ).to(device)

    logger.info(f"SRN parameters: {count_parameters(srn):,}")
    logger.info(f"SRN trainable parameters: {count_parameters(srn, only_trainable=True):,}")

    # CLIP projection for quality scoring (fixed random projection)
    clip_projection = nn.Linear(1024, 512, bias=False).to(device)
    for param in clip_projection.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    lr = train_cfg.get("lr_phase2", 3e-4)
    total_steps = train_cfg.get("phase2_steps", 50000)

    optimizer = torch.optim.Adam(
        srn.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    num_frames = model_cfg.get("num_frames", 16)
    latent_channels = vae_cfg.get("latent_channels", 4)
    latent_hw = model_cfg.get("latent_size", 32)
    K = 8  # Candidates per sentence

    dataset = SignLanguageVideoDataset(
        data_root=args.data_dir,
        datasets=data_cfg.get("datasets", ["PHOENIX14T", "How2Sign", "USTC-CSL"]),
        split="train",
        num_frames=data_cfg.get("num_frames", 32),
        resolution=data_cfg.get("resolution", 256),
        text_emb_dim=data_cfg.get("text_emb_dim", 1024),
        max_text_length=data_cfg.get("max_text_length", 77),
    )

    batch_size = 1  # One sentence at a time (K candidates per sentence)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
    )

    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Candidates per sentence (K): {K}")
    logger.info(f"Total training steps: {total_steps}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_interval = train_cfg.get("log_interval", 100)
    save_interval = train_cfg.get("save_interval", 5000)

    srn.train()
    global_step = 0
    running_loss = 0.0
    log_count = 0
    data_iter = iter(dataloader)

    logger.info("Starting Phase 2 training...")
    start_time = time.time()

    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        video_ref, gloss, text_emb = batch
        video_ref = video_ref.squeeze(0)  # (T, 3, H, W)
        text_emb = text_emb.squeeze(0)    # (L, D)

        # Subsample frames for diffusion model
        T_full = video_ref.shape[0]
        if T_full > num_frames:
            indices = torch.linspace(0, T_full - 1, num_frames, dtype=torch.long)
            video_ref_sub = video_ref[indices]
        else:
            video_ref_sub = video_ref

        # Generate K candidates using frozen diffusion model
        candidates = generate_candidates(
            unet, vae, scheduler, text_emb,
            K=K,
            num_frames=num_frames,
            latent_channels=latent_channels,
            latent_hw=latent_hw,
            device=device,
        )

        # Compute quality scores and rankings
        quality_scores, rankings = compute_quality_scores(
            candidates, video_ref_sub, text_emb,
            clip_projection,
            num_joints=srn_cfg.get("num_joints", 148),
            device=device,
        )
        rankings = rankings.to(device)

        # Forward all candidates through SRN to get predicted rewards
        srn_rewards = []
        for candidate in candidates:
            v = candidate.unsqueeze(0).to(device)  # (1, T, 3, H, W)
            T_cand = v.shape[1]

            # Estimate keypoints for the candidate
            kpts = estimate_keypoints_from_video(
                v.squeeze(0), srn_cfg.get("num_joints", 148)
            ).unsqueeze(0).to(device)

            # Text embedding (512-d for SRN)
            text_512 = text_emb.mean(dim=0)[:512].unsqueeze(0).to(device)
            if text_512.shape[-1] < 512:
                text_512 = F.pad(text_512, (0, 512 - text_512.shape[-1]))

            reward = srn(v, kpts, text_512)  # (1,)
            srn_rewards.append(reward)

        srn_rewards_tensor = torch.cat(srn_rewards, dim=0)  # (K,)

        # Listwise ranking loss
        loss = listwise_ranking_loss(
            srn_rewards_tensor.unsqueeze(0),  # (1, K)
            rankings.unsqueeze(0).long(),      # (1, K)
            temperature=1.0,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(srn.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        running_loss += loss.item()
        log_count += 1

        # Logging
        if global_step % log_interval == 0:
            avg_loss = running_loss / log_count
            elapsed = time.time() - start_time
            steps_per_sec = global_step / max(elapsed, 1.0)

            logger.info(
                f"Step {global_step}/{total_steps} | "
                f"Loss: {avg_loss:.6f} | "
                f"SRN Rewards: [{', '.join(f'{r:.3f}' for r in srn_rewards_tensor.detach().cpu().tolist())}] | "
                f"Rankings: [{', '.join(f'{int(r)}' for r in rankings.tolist())}] | "
                f"Steps/s: {steps_per_sec:.2f}"
            )

            running_loss = 0.0
            log_count = 0

        # Save checkpoint
        if global_step % save_interval == 0:
            ckpt_path = str(output_dir / f"checkpoint_{global_step}.pt")
            save_checkpoint(
                model=srn,
                optimizer=optimizer,
                epoch=0,
                step=global_step,
                path=ckpt_path,
                extra={"config": config},
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = str(output_dir / "final.pt")
    save_checkpoint(
        model=srn,
        optimizer=optimizer,
        epoch=0,
        step=global_step,
        path=final_path,
        extra={"config": config},
    )
    logger.info(f"Phase 2 training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
