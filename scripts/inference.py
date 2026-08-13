#!/usr/bin/env python3
"""
Inference: Sign Language Video Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates sign-language videos from text prompts using the trained
SignRL-Diff pipeline.  The script loads the UNet (with LoRA adapters),
VAE, and Policy Network, then runs the guided denoising loop.

Pipeline:
    1. Encode text prompt into CLIP-style embedding
    2. Sample initial noise z_K ~ N(0, I)
    3. For k = K-1 down to 0:
       a. UNet predicts noise: eps_hat_k = UNet(z_k, k, c)
       b. Policy Network produces correction: a_k = pi(z_k, k, c, eps_hat_k)
       c. Apply correction: eps_tilde_k = eps_hat_k + correction(a_k)
       d. Scheduler step: z_{k-1} = step(eps_tilde_k, k, z_k)
    4. Decode z_0 via VAE: video = VAE.decode(z_0)
    5. Save as MP4 or GIF

Usage:
    python -m signrl_diff.scripts.inference \
        --text "HELLO WORLD HOW ARE YOU" \
        --checkpoint_dir ./checkpoints \
        --output output.mp4 \
        --num_frames 32
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signrl_diff.models.diffusion import VideoUNet, AutoencoderKL, DDPMScheduler
from signrl_diff.models.rl import PolicyNetwork
from signrl_diff.utils import set_seed, count_parameters
from signrl_diff.utils.helpers import action_to_correction, build_hand_sparse_projection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference")


# ======================================================================
# Text Encoding
# ======================================================================

class SimpleTextEncoder(nn.Module):
    """Deterministic text encoder that produces consistent embeddings.

    Maps text strings to fixed-dimensional embeddings using character-level
    hashing and a learned MLP.  This provides a lightweight alternative
    when a full CLIP model is not available.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimensionality.
    max_length : int
        Maximum token sequence length.
    """

    def __init__(self, embed_dim: int = 1024, max_length: int = 77) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.char_embed = nn.Embedding(256, 128)
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )

    def forward(self, text: str) -> torch.Tensor:
        """Encode a text string into an embedding tensor.

        Parameters
        ----------
        text : str
            Input text string.

        Returns
        -------
        Tensor, shape ``(max_length, embed_dim)``
        """
        chars = [ord(c) % 256 for c in text]
        if len(chars) < self.max_length:
            chars = chars + [0] * (self.max_length - len(chars))
        else:
            chars = chars[:self.max_length]

        char_tensor = torch.tensor(chars, dtype=torch.long,
                                   device=next(self.parameters()).device)
        char_feats = self.char_embed(char_tensor)  # (max_length, 128)
        embedding = self.encoder(char_feats)       # (max_length, embed_dim)
        return embedding


# ======================================================================
# Video Saving Utilities
# ======================================================================

def save_video_mp4(
    video_tensor: torch.Tensor,
    output_path: str,
    fps: int = 16,
) -> None:
    """Save a video tensor as MP4.

    Parameters
    ----------
    video_tensor : Tensor, shape ``(T, 3, H, W)``
        Video tensor in [-1, 1] range.
    output_path : str
        Output file path.
    fps : int
        Frames per second.
    """
    import cv2

    T, C, H, W = video_tensor.shape
    frames_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()
    frames_np = ((frames_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for frame in frames_np:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    logger.info(f"Video saved to {output_path}")


def save_video_gif(
    video_tensor: torch.Tensor,
    output_path: str,
    fps: int = 16,
) -> None:
    """Save a video tensor as animated GIF.

    Parameters
    ----------
    video_tensor : Tensor, shape ``(T, 3, H, W)``
        Video tensor in [-1, 1] range.
    output_path : str
        Output file path.
    fps : int
        Frames per second.
    """
    from PIL import Image

    T, C, H, W = video_tensor.shape
    frames_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()
    frames_np = ((frames_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

    pil_frames = [Image.fromarray(frame) for frame in frames_np]
    duration_ms = int(1000.0 / fps)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    logger.info(f"GIF saved to {output_path}")


def save_frames_as_images(
    video_tensor: torch.Tensor,
    output_dir: str,
) -> None:
    """Save individual frames as PNG images.

    Parameters
    ----------
    video_tensor : Tensor, shape ``(T, 3, H, W)``
        Video tensor in [-1, 1] range.
    output_dir : str
        Output directory path.
    """
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    T, C, H, W = video_tensor.shape
    frames_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()
    frames_np = ((frames_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)

    for i, frame in enumerate(frames_np):
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f"frame_{i:04d}.png"))

    logger.info(f"Frames saved to {output_dir}")


# ======================================================================
# Inference Pipeline
# ======================================================================

@torch.no_grad()
def generate_video(
    unet: nn.Module,
    vae: nn.Module,
    scheduler: nn.Module,
    policy: nn.Module,
    text_embedding: torch.Tensor,
    K: int = 50,
    num_frames: int = 16,
    latent_channels: int = 4,
    latent_hw: int = 32,
    use_policy: bool = True,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a video from a text embedding using guided denoising.

    Parameters
    ----------
    unet : nn.Module
        Diffusion UNet (with LoRA adapters loaded).
    vae : nn.Module
        Video VAE decoder.
    scheduler : nn.Module
        DDPM scheduler.
    policy : nn.Module
        Policy network for corrections.
    text_embedding : Tensor, shape ``(1, L, D)``
        Text condition.
    K : int
        Number of denoising steps.
    num_frames : int
        Temporal frames in latent.
    latent_channels : int
    latent_hw : int
    use_policy : bool
        Whether to apply policy corrections.
    device : torch.device

    Returns
    -------
    Tensor, shape ``(T, 3, H, W)``
        Generated video tensor in [-1, 1].
    """
    # Sample initial noise
    z_k = torch.randn(
        1, num_frames, latent_channels, latent_hw, latent_hw,
        device=device,
    )

    # Compute stride for mapping K steps to scheduler's full T steps
    T_total = scheduler.num_train_steps
    if K < T_total:
        # Subsample timesteps: use every (T_total // K) steps
        timestep_indices = torch.linspace(T_total - 1, 0, K, dtype=torch.long)
    else:
        timestep_indices = torch.arange(T_total - 1, -1, -1, dtype=torch.long)
        K = T_total

    # Build action projection layers for policy corrections
    latent_flat = num_frames * latent_channels * latent_hw * latent_hw
    w_global = nn.Linear(64, latent_flat, bias=False).to(device)
    nn.init.normal_(w_global.weight, mean=0.0, std=0.01)

    m_hand_left = build_hand_sparse_projection(
        num_frames=num_frames,
        latent_channels=latent_channels,
        latent_hw=latent_hw,
        hand_action_dim=32,
        hand_region_ratio=0.25,
        device=device,
    )
    m_hand_right = build_hand_sparse_projection(
        num_frames=num_frames,
        latent_channels=latent_channels,
        latent_hw=latent_hw,
        hand_action_dim=32,
        hand_region_ratio=0.25,
        device=device,
    )

    logger.info(f"Running denoising loop: {K} steps...")

    for step_idx in range(K):
        t = timestep_indices[step_idx].item()
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

        # UNet predicts noise
        eps_hat_k = unet(z_k, t_tensor, text_embedding)

        # Policy Network correction
        if use_policy and policy is not None:
            state = (z_k, t_tensor, eps_hat_k)
            action, _ = policy(state, text_embedding)

            # Apply correction
            eps_corr = action_to_correction(
                action, w_global, m_hand_left, m_hand_right,
                num_frames=num_frames,
                latent_channels=latent_channels,
                latent_hw=latent_hw,
            )
            eps_hat_k = eps_hat_k + eps_corr

        # Scheduler reverse step
        z_k = scheduler.step(eps_hat_k, t, z_k)

    # Decode final latent to video
    if hasattr(vae, "decode"):
        video = vae.decode(z_k)
    else:
        B, T, C, H, W = z_k.shape
        z_flat = z_k.reshape(B * T, C, H, W)
        v_flat = vae(z_flat)
        _, Co, Ho, Wo = v_flat.shape
        video = v_flat.reshape(B, T, Co, Ho, Wo)

    video = video.squeeze(0)  # (T, 3, H, W)
    video = video.clamp(-1.0, 1.0)

    return video


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="SignRL-Diff Inference")
    parser.add_argument("--text", type=str, required=True,
                        help="Input text sentence for sign language generation")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory containing trained checkpoints")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Output file path (.mp4 or .gif)")
    parser.add_argument("--num_frames", type=int, default=32,
                        help="Number of frames in output video")
    parser.add_argument("--denoising_steps", type=int, default=50,
                        help="Number of denoising steps (K)")
    parser.add_argument("--fps", type=int, default=16,
                        help="Frames per second for output video")
    parser.add_argument("--no_policy", action="store_true",
                        help="Disable policy network corrections")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_frames", action="store_true",
                        help="Also save individual frames as PNG")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)

    # Load config (try multiple locations)
    config = {}
    for config_path in [
        checkpoint_dir / "config.yaml",
        checkpoint_dir / "phase1" / "config.yaml",
        Path("configs/default.yaml"),
    ]:
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            break

    model_cfg = config.get("model", {})
    vae_cfg = config.get("vae", {})
    diff_cfg = config.get("diffusion", {})

    num_frames_latent = model_cfg.get("num_frames", 16)
    latent_channels = vae_cfg.get("latent_channels", 4)
    latent_hw = model_cfg.get("latent_size", 32)
    text_dim = model_cfg.get("text_dim", 1024)

    # ------------------------------------------------------------------
    # Build models
    # ------------------------------------------------------------------
    unet = VideoUNet(
        in_channels=latent_channels,
        channel_config=model_cfg.get("unet_channels", [128, 256, 512, 512]),
        text_dim=text_dim,
        cond_dim=model_cfg.get("cond_dim", 512),
        num_heads=model_cfg.get("num_heads", 8),
    )

    vae = AutoencoderKL(
        latent_channels=latent_channels,
        base_channels=vae_cfg.get("base_channels", 64),
        kl_weight=vae_cfg.get("kl_weight", 1e-4),
    )

    scheduler = DDPMScheduler(
        num_train_steps=diff_cfg.get("num_train_steps", 1000),
        beta_schedule=diff_cfg.get("beta_schedule", "linear"),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
    )

    policy = PolicyNetwork(
        num_frames=num_frames_latent,
        latent_channels=latent_channels,
        latent_hw=latent_hw,
        text_dim=text_dim,
        hidden_dim=512,
        num_heads=model_cfg.get("num_heads", 8),
    )

    # ------------------------------------------------------------------
    # Load checkpoints
    # ------------------------------------------------------------------
    # Phase 1: UNet + VAE
    phase1_paths = [
        checkpoint_dir / "phase1" / "final.pt",
        checkpoint_dir / "phase1" / "latest.pt",
        checkpoint_dir / "unet.pt",
    ]
    for p in phase1_paths:
        if p.exists():
            ckpt = torch.load(str(p), map_location=device, weights_only=False)
            unet.load_state_dict(ckpt["model_state_dict"])
            if "vae_state_dict" in ckpt:
                vae.load_state_dict(ckpt["vae_state_dict"])
            logger.info(f"Loaded UNet/VAE from {p}")
            break
    else:
        logger.warning("No Phase 1 checkpoint found, using random UNet/VAE")

    # Inject LoRA (architecture must match training)
    lora_rank = model_cfg.get("lora_rank", 16)
    lora_alpha = model_cfg.get("lora_alpha", 16.0)
    unet.inject_lora(rank=lora_rank, alpha=lora_alpha)

    # Load LoRA weights from Phase 3
    lora_paths = [
        checkpoint_dir / "phase3" / "lora_final.pt",
        checkpoint_dir / "phase3" / "lora.pt",
    ]
    for p in lora_paths:
        if p.exists():
            lora_ckpt = torch.load(str(p), map_location=device, weights_only=False)
            lora_state = lora_ckpt.get("lora_state", {})
            unet_module = unet
            for name, param in unet_module.named_parameters():
                if name in lora_state:
                    param.data.copy_(lora_state[name])
            logger.info(f"Loaded LoRA weights from {p}")
            break
    else:
        logger.warning("No LoRA checkpoint found, using untrained LoRA")

    # Merge LoRA into base weights for efficient inference
    unet.merge_lora()

    # Phase 3: Policy Network
    policy_paths = [
        checkpoint_dir / "phase3" / "final.pt",
        checkpoint_dir / "phase3" / "latest.pt",
        checkpoint_dir / "policy.pt",
    ]
    for p in policy_paths:
        if p.exists():
            ckpt = torch.load(str(p), map_location=device, weights_only=False)
            if "policy_state_dict" in ckpt:
                policy.load_state_dict(ckpt["policy_state_dict"])
                logger.info(f"Loaded Policy Network from {p}")
                break
    else:
        logger.warning("No Policy Network checkpoint found, using random policy")

    # Move to device and set eval mode
    unet = unet.to(device)
    vae = vae.to(device)
    scheduler = scheduler.to(device)
    policy = policy.to(device)

    unet.eval()
    vae.eval()
    policy.eval()

    logger.info(f"UNet parameters: {count_parameters(unet):,}")
    logger.info(f"Policy parameters: {count_parameters(policy):,}")

    # ------------------------------------------------------------------
    # Encode text
    # ------------------------------------------------------------------
    text_encoder = SimpleTextEncoder(
        embed_dim=text_dim, max_length=77
    ).to(device)
    text_encoder.eval()

    text_embedding = text_encoder(args.text)  # (77, text_dim)
    text_embedding = text_embedding.unsqueeze(0)  # (1, 77, text_dim)

    logger.info(f"Text prompt: '{args.text}'")
    logger.info(f"Text embedding shape: {text_embedding.shape}")

    # ------------------------------------------------------------------
    # Generate video
    # ------------------------------------------------------------------
    K = args.denoising_steps
    use_policy = not args.no_policy

    video = generate_video(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        policy=policy,
        text_embedding=text_embedding,
        K=K,
        num_frames=num_frames_latent,
        latent_channels=latent_channels,
        latent_hw=latent_hw,
        use_policy=use_policy,
        device=device,
    )

    # Temporal interpolation to target frame count
    T_gen = video.shape[0]
    if T_gen < args.num_frames:
        indices = torch.linspace(0, T_gen - 1, args.num_frames)
        idx_floor = indices.floor().long().clamp(max=T_gen - 2)
        idx_ceil = (idx_floor + 1).clamp(max=T_gen - 1)
        alpha = (indices - idx_floor.float()).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha = alpha.to(video.device)
        video = video[idx_floor] * (1 - alpha) + video[idx_ceil] * alpha
    elif T_gen > args.num_frames:
        indices = torch.linspace(0, T_gen - 1, args.num_frames, dtype=torch.long)
        video = video[indices]

    logger.info(f"Generated video shape: {video.shape}")

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    output_path = args.output
    output_ext = Path(output_path).suffix.lower()

    if output_ext == ".gif":
        save_video_gif(video, output_path, fps=args.fps)
    elif output_ext in (".mp4", ".avi", ".mov"):
        save_video_mp4(video, output_path, fps=args.fps)
    else:
        save_video_mp4(video, output_path + ".mp4", fps=args.fps)

    if args.save_frames:
        frames_dir = str(Path(output_path).with_suffix("")) + "_frames"
        save_frames_as_images(video, frames_dir)

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
