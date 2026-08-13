#!/usr/bin/env python3
"""
Phase 1: Diffusion Pre-training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-trains the video diffusion model (VideoUNet + AutoencoderKL) on sign
language datasets.  The VAE encoder maps video clips into a latent space;
the UNet learns to predict Gaussian noise added at random diffusion
timesteps.

Training objective:
    L = E_{t, x_0, eps} [ || eps - UNet(z_t, t, c) ||^2 ]

where z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * eps,
and z_0 = VAE.encode(video).

Usage:
    python -m signrl_diff.scripts.train_phase1 \
        --config configs/default.yaml \
        --data_dir ./data \
        --output_dir ./checkpoints/phase1 \
        --resume ./checkpoints/phase1/latest.pt
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from signrl_diff.models.diffusion import VideoUNet, AutoencoderKL, DDPMScheduler
from signrl_diff.data import SignLanguageVideoDataset
from signrl_diff.utils import set_seed, count_parameters, save_checkpoint, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_phase1")


# ======================================================================
# EMA (Exponential Moving Average) Model
# ======================================================================

class EMAModel:
    """Maintains an exponential moving average of model parameters.

    Parameters
    ----------
    model : nn.Module
        The source model whose parameters are tracked.
    decay : float
        Decay rate (close to 1.0 for slow averaging).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with EMA of current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> None:
        """Copy shadow parameters into the model (for evaluation/saving)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


# ======================================================================
# Cosine Annealing LR Scheduler
# ======================================================================

def build_cosine_annealing_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 5000,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a cosine annealing scheduler with linear warmup.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    total_steps : int
        Total number of training steps.
    warmup_steps : int
        Number of linear warmup steps.

    Returns
    -------
    LambdaLR scheduler.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ======================================================================
# Training Step
# ======================================================================

def train_step(
    batch: tuple,
    vae: nn.Module,
    unet: nn.Module,
    scheduler: nn.Module,
    device: torch.device,
    num_frames: int,
) -> Dict[str, float]:
    """Execute a single training step.

    Parameters
    ----------
    batch : tuple
        (video, gloss, text_embedding) from the dataset.
    vae : nn.Module
        Frozen VAE encoder.
    unet : nn.Module
        Trainable UNet.
    scheduler : nn.Module
        DDPM scheduler.
    device : torch.device
    num_frames : int
        Number of frames for the diffusion model.

    Returns
    -------
    dict with loss statistics.
    """
    video_full, glosses, text_embeddings = batch

    video_full = video_full.to(device)
    text_embeddings = text_embeddings.to(device)

    B = video_full.shape[0]

    # Subsample frames if video has more than num_frames
    T_full = video_full.shape[1]
    if T_full > num_frames:
        indices = torch.linspace(0, T_full - 1, num_frames, dtype=torch.long)
        video = video_full[:, indices]
    else:
        video = video_full

    # Encode video to latent space (VAE is frozen, no gradient)
    with torch.no_grad():
        z_0, mu, logvar = vae.encode(video)

    # Sample random timesteps
    t = torch.randint(0, scheduler.num_train_steps, (B,), device=device, dtype=torch.long)

    # Add noise to latents
    noise = torch.randn_like(z_0)
    z_t, noise_used = scheduler.add_noise(z_0, t, noise)

    # UNet predicts noise
    noise_pred = unet(z_t, t, text_embeddings)

    # MSE loss
    loss = F.mse_loss(noise_pred, noise_used)

    # Compute per-noise-level MSE for logging (bucket timesteps into bins)
    with torch.no_grad():
        per_level_mse = F.mse_loss(noise_pred, noise_used, reduction="none")
        per_level_mse = per_level_mse.mean(dim=list(range(1, per_level_mse.dim())))
        # Group into 10 bins
        bin_size = scheduler.num_train_steps // 10
        bin_mse = {}
        for bin_idx in range(10):
            low = bin_idx * bin_size
            high = (bin_idx + 1) * bin_size
            mask = (t >= low) & (t < high)
            if mask.any():
                bin_mse[f"mse_bin_{bin_idx}"] = per_level_mse[mask].mean().item()

    return {
        "loss": loss,
        "mse": loss.item(),
        **bin_mse,
    }


# ======================================================================
# Main Training Loop
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Diffusion Pre-training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root directory for datasets")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase1",
                        help="Directory for saving checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
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
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build models
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

    # Freeze VAE (only UNet is trained in Phase 1)
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()

    unet = unet.to(device)
    vae = vae.to(device)
    scheduler = scheduler.to(device)

    logger.info(f"UNet parameters: {count_parameters(unet):,}")
    logger.info(f"UNet trainable parameters: {count_parameters(unet, only_trainable=True):,}")
    logger.info(f"VAE parameters: {count_parameters(vae):,} (frozen)")

    # ------------------------------------------------------------------
    # DataParallel placeholder for multi-GPU
    # ------------------------------------------------------------------
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logger.info(f"Using DataParallel with {num_gpus} GPUs")
        unet = nn.DataParallel(unet)
        vae = nn.DataParallel(vae)

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    lr = train_cfg.get("lr_phase1", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    total_steps = train_cfg.get("phase1_steps", 500000)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    lr_scheduler = build_cosine_annealing_scheduler(
        optimizer, total_steps, warmup_steps=5000,
    )

    # ------------------------------------------------------------------
    # EMA model
    # ------------------------------------------------------------------
    ema_model = EMAModel(unet, decay=0.9999)

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    if args.resume and Path(args.resume).exists():
        meta = load_checkpoint(args.resume, unet, optimizer, device=device)
        start_step = meta.get("step", 0)
        if "ema_state_dict" in meta:
            ema_model.shadow = meta["ema_state_dict"]
        lr_scheduler.last_epoch = start_step
        logger.info(f"Resumed from step {start_step}")

    # ------------------------------------------------------------------
    # Dataset and DataLoader
    # ------------------------------------------------------------------
    num_frames = model_cfg.get("num_frames", 16)
    dataset = SignLanguageVideoDataset(
        data_root=args.data_dir,
        datasets=data_cfg.get("datasets", ["PHOENIX14T", "How2Sign", "USTC-CSL"]),
        split="train",
        num_frames=data_cfg.get("num_frames", 32),
        resolution=data_cfg.get("resolution", 256),
        text_emb_dim=data_cfg.get("text_emb_dim", 1024),
        max_text_length=data_cfg.get("max_text_length", 77),
    )

    batch_size = train_cfg.get("batch_size", 4)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
    )

    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Total training steps: {total_steps}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_interval = train_cfg.get("log_interval", 100)
    save_interval = 10000  # Save every 10K steps per spec
    grad_clip = train_cfg.get("grad_clip", 1.0)

    unet.train()
    global_step = start_step
    running_loss = 0.0
    running_grad_norm = 0.0
    epoch = 0
    log_count = 0

    data_iter = iter(dataloader)

    logger.info("Starting Phase 1 training...")
    start_time = time.time()

    while global_step < total_steps:
        # Get next batch (cycle through dataloader)
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Training step
        stats = train_step(batch, vae, unet, scheduler, device, num_frames)
        loss = stats["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping and norm logging
        grad_norm = nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, unet.parameters()),
            grad_clip,
        )

        optimizer.step()
        lr_scheduler.step()

        # Update EMA
        ema_model.update(unet)

        # Accumulate stats
        running_loss += stats["mse"]
        running_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        global_step += 1
        log_count += 1

        # Logging
        if global_step % log_interval == 0:
            avg_loss = running_loss / log_count
            avg_grad_norm = running_grad_norm / log_count
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            steps_per_sec = global_step / max(elapsed, 1.0)

            logger.info(
                f"Step {global_step}/{total_steps} | "
                f"Loss: {avg_loss:.6f} | "
                f"Grad Norm: {avg_grad_norm:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Steps/s: {steps_per_sec:.2f} | "
                f"Epoch: {epoch}"
            )

            # Log per-bin MSE when available
            for key in sorted(stats.keys()):
                if key.startswith("mse_bin_"):
                    logger.info(f"  {key}: {stats[key]:.6f}")

            running_loss = 0.0
            running_grad_norm = 0.0
            log_count = 0

        # Save checkpoint
        if global_step % save_interval == 0:
            unet_module = unet.module if hasattr(unet, "module") else unet
            ckpt_path = str(output_dir / f"checkpoint_{global_step}.pt")

            save_checkpoint(
                model=unet_module,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                path=ckpt_path,
                extra={
                    "ema_state_dict": ema_model.shadow,
                    "config": config,
                },
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

            # Also save a "latest" symlink-style checkpoint
            latest_path = str(output_dir / "latest.pt")
            save_checkpoint(
                model=unet_module,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                path=latest_path,
                extra={
                    "ema_state_dict": ema_model.shadow,
                    "config": config,
                },
            )

    # Final save
    unet_module = unet.module if hasattr(unet, "module") else unet
    final_path = str(output_dir / "final.pt")
    save_checkpoint(
        model=unet_module,
        optimizer=optimizer,
        epoch=epoch,
        step=global_step,
        path=final_path,
        extra={
            "ema_state_dict": ema_model.shadow,
            "config": config,
        },
    )
    logger.info(f"Phase 1 training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
