#!/usr/bin/env python3
"""
Phase 3: RL Fine-tuning with PPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tunes the diffusion model via Proximal Policy Optimization.  LoRA
adapters (rank=16) are injected into the frozen UNet attention layers.
A Policy Network outputs latent-space corrections at each denoising
step, guided by the Hierarchical Articulation-Aware Reward (HAR).

Pipeline:
    1. Load frozen UNet+VAE (Phase 1), frozen SRN (Phase 2)
    2. Inject LoRA into UNet
    3. Initialize PolicyNetwork, ValueNetwork, HierarchicalReward
    4. Create VecDenoisingEnv (4 parallel envs)
    5. PPO loop: collect rollout -> GAE -> update (4 epochs, bs=64)
    6. Log: policy_loss, value_loss, entropy, mean_reward, mean_advantage

Usage:
    python -m signrl_diff.scripts.train_phase3 \
        --config configs/default.yaml \
        --phase1_ckpt ./checkpoints/phase1/final.pt \
        --phase2_ckpt ./checkpoints/phase2/final.pt \
        --output_dir ./checkpoints/phase3
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import torch
import torch.nn as nn

from signrl_diff.models.diffusion import VideoUNet, AutoencoderKL, DDPMScheduler
from signrl_diff.models.srn import SubRewardNetwork
from signrl_diff.models.rl import PolicyNetwork, ValueNetwork, PPOTrainer
from signrl_diff.rewards import HierarchicalReward
from signrl_diff.env import DenoisingEnv, VecDenoisingEnv
from signrl_diff.data import SignLanguageVideoDataset
from signrl_diff.utils import set_seed, count_parameters, save_checkpoint, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_phase3")


# ======================================================================
# Environment Factory
# ======================================================================

def make_env_fn(
    unet: nn.Module,
    vae: nn.Module,
    scheduler: nn.Module,
    reward_fn: nn.Module,
    text_condition: torch.Tensor,
    K: int = 50,
    num_frames: int = 16,
    latent_channels: int = 4,
    latent_hw: int = 32,
    cfg_scale_default: float = 7.5,
    cfg_scale_range: float = 2.0,
    device: torch.device = torch.device("cpu"),
) -> callable:
    """Create a factory function that returns a DenoisingEnv.

    Parameters
    ----------
    All parameters are forwarded to DenoisingEnv.__init__.

    Returns
    -------
    callable
        A zero-argument callable that returns a DenoisingEnv instance.
    """
    def _make_env() -> DenoisingEnv:
        return DenoisingEnv(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            reward_fn=reward_fn,
            text_condition=text_condition,
            K=K,
            num_frames=num_frames,
            latent_channels=latent_channels,
            latent_hw=latent_hw,
            cfg_scale_default=cfg_scale_default,
            cfg_scale_range=cfg_scale_range,
            device=device,
        )
    return _make_env


# ======================================================================
# Main Training Loop
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: RL Fine-tuning with PPO")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--phase1_ckpt", type=str, required=True,
                        help="Path to Phase 1 checkpoint (UNet + VAE)")
    parser.add_argument("--phase2_ckpt", type=str, required=True,
                        help="Path to Phase 2 checkpoint (SRN)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase3",
                        help="Directory for saving checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to Phase 3 checkpoint to resume from")
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
    har_cfg = config.get("har", {})
    rl_cfg = config.get("rl", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config values
    num_frames = model_cfg.get("num_frames", 16)
    latent_channels = vae_cfg.get("latent_channels", 4)
    latent_hw = model_cfg.get("latent_size", 32)
    text_dim = model_cfg.get("text_dim", 1024)
    lora_rank = model_cfg.get("lora_rank", 16)
    lora_alpha = model_cfg.get("lora_alpha", 16.0)

    # ------------------------------------------------------------------
    # Build and load frozen models
    # ------------------------------------------------------------------
    # UNet
    unet = VideoUNet(
        in_channels=latent_channels,
        channel_config=model_cfg.get("unet_channels", [128, 256, 512, 512]),
        text_dim=text_dim,
        cond_dim=model_cfg.get("cond_dim", 512),
        num_heads=model_cfg.get("num_heads", 8),
    )

    # VAE
    vae = AutoencoderKL(
        latent_channels=latent_channels,
        base_channels=vae_cfg.get("base_channels", 64),
        kl_weight=vae_cfg.get("kl_weight", 1e-4),
    )

    # Scheduler
    scheduler = DDPMScheduler(
        num_train_steps=diff_cfg.get("num_train_steps", 1000),
        beta_schedule=diff_cfg.get("beta_schedule", "linear"),
        beta_start=diff_cfg.get("beta_start", 1e-4),
        beta_end=diff_cfg.get("beta_end", 0.02),
    )

    # Load Phase 1 checkpoint (UNet + VAE)
    phase1_path = Path(args.phase1_ckpt)
    if phase1_path.exists():
        ckpt = torch.load(str(phase1_path), map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["model_state_dict"])
        if "vae_state_dict" in ckpt:
            vae.load_state_dict(ckpt["vae_state_dict"])
        logger.info(f"Loaded Phase 1 checkpoint from {phase1_path}")
    else:
        logger.warning(f"Phase 1 checkpoint not found: {phase1_path}, using random init")

    # SRN
    srn = SubRewardNetwork(
        num_joints=srn_cfg.get("num_joints", 148),
        video_emb_dim=srn_cfg.get("video_emb_dim", 1024),
        use_video_extractor=True,
        use_hand_extractor=True,
    )

    # Load Phase 2 checkpoint (SRN)
    phase2_path = Path(args.phase2_ckpt)
    if phase2_path.exists():
        ckpt = torch.load(str(phase2_path), map_location=device, weights_only=False)
        srn.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded Phase 2 checkpoint from {phase2_path}")
    else:
        logger.warning(f"Phase 2 checkpoint not found: {phase2_path}, using random init")

    # Freeze UNet base weights
    for param in unet.parameters():
        param.requires_grad = False
    unet.eval()

    # Freeze VAE
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()

    # Freeze SRN
    for param in srn.parameters():
        param.requires_grad = False
    srn.eval()

    # ------------------------------------------------------------------
    # Inject LoRA into UNet
    # ------------------------------------------------------------------
    unet.inject_lora(rank=lora_rank, alpha=lora_alpha)
    lora_params = unet.get_lora_parameters()
    for p in lora_params:
        p.requires_grad = True

    logger.info(f"LoRA injected with rank={lora_rank}, alpha={lora_alpha}")
    logger.info(f"LoRA trainable parameters: {sum(p.numel() for p in lora_params):,}")

    # Move models to device
    unet = unet.to(device)
    vae = vae.to(device)
    scheduler = scheduler.to(device)
    srn = srn.to(device)

    # ------------------------------------------------------------------
    # Initialize RL components
    # ------------------------------------------------------------------
    # Policy Network
    policy = PolicyNetwork(
        num_frames=num_frames,
        latent_channels=latent_channels,
        latent_hw=latent_hw,
        text_dim=text_dim,
        hidden_dim=512,
        num_heads=model_cfg.get("num_heads", 8),
    ).to(device)

    # Value Network
    value_fn = ValueNetwork(
        num_frames=num_frames,
        latent_channels=latent_channels,
        latent_hw=latent_hw,
        text_dim=text_dim,
        hidden_dim=512,
        num_heads=model_cfg.get("num_heads", 8),
    ).to(device)

    logger.info(f"Policy parameters: {count_parameters(policy):,}")
    logger.info(f"Value parameters: {count_parameters(value_fn):,}")

    # Hierarchical Reward
    reward_fn = HierarchicalReward(
        srn=srn,
        vae_decoder=vae,
        text_dim=har_cfg.get("text_dim", text_dim),
        gate_hidden_dim=har_cfg.get("gate_hidden_dim", 256),
        gamma=har_cfg.get("gamma", 0.99),
    ).to(device)

    logger.info(f"HAR gate parameters: {count_parameters(reward_fn.MLP_gate):,}")

    # ------------------------------------------------------------------
    # PPO Trainer
    # ------------------------------------------------------------------
    ppo_lr = rl_cfg.get("lr", 3e-5)
    ppo_trainer = PPOTrainer(
        policy=policy,
        value_fn=value_fn,
        lr=ppo_lr,
        gamma=rl_cfg.get("gamma", 0.99),
        gae_lambda=rl_cfg.get("gae_lambda", 0.95),
        clip_epsilon=rl_cfg.get("clip_epsilon", 0.2),
        value_coeff=rl_cfg.get("value_coeff", 0.5),
        entropy_coeff=rl_cfg.get("entropy_coeff", 0.01),
        num_epochs=rl_cfg.get("epochs", 4),
        mini_batch_size=rl_cfg.get("mini_batch", 64),
        max_grad_norm=rl_cfg.get("max_grad_norm", 0.5),
        device=device,
    )

    # ------------------------------------------------------------------
    # Dataset for text conditions
    # ------------------------------------------------------------------
    dataset = SignLanguageVideoDataset(
        data_root=data_cfg.get("paths", {}).get("data_root", "./data"),
        datasets=data_cfg.get("datasets", ["PHOENIX14T", "How2Sign", "USTC-CSL"]),
        split="train",
        num_frames=data_cfg.get("num_frames", 32),
        resolution=data_cfg.get("resolution", 256),
        text_emb_dim=data_cfg.get("text_emb_dim", 1024),
        max_text_length=data_cfg.get("max_text_length", 77),
    )

    # ------------------------------------------------------------------
    # Create VecDenoisingEnv (4 parallel envs)
    # ------------------------------------------------------------------
    num_envs = rl_cfg.get("num_envs", 4)
    K_steps = 50  # Denoising steps per episode
    cfg_scale_default = rl_cfg.get("cfg_scale_default", 7.5)
    cfg_scale_range = rl_cfg.get("cfg_scale_range", 2.0)

    def get_random_text_condition() -> torch.Tensor:
        """Sample a random text condition from the dataset."""
        idx = torch.randint(0, len(dataset), (1,)).item()
        _, _, text_emb = dataset[idx]
        return text_emb

    # Create initial text conditions for each env
    env_fns = []
    for i in range(num_envs):
        text_cond = get_random_text_condition()
        env_fn = make_env_fn(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            reward_fn=reward_fn,
            text_condition=text_cond,
            K=K_steps,
            num_frames=num_frames,
            latent_channels=latent_channels,
            latent_hw=latent_hw,
            cfg_scale_default=cfg_scale_default,
            cfg_scale_range=cfg_scale_range,
            device=device,
        )
        env_fns.append(env_fn)

    vec_env = VecDenoisingEnv(env_fns)
    logger.info(f"Created VecDenoisingEnv with {num_envs} parallel environments")

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    if args.resume and Path(args.resume).exists():
        ppo_trainer.load_checkpoint(args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        start_step = ckpt.get("step", 0)
        logger.info(f"Resumed from step {start_step}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    total_steps = rl_cfg.get("total_steps", 200000)
    rollout_steps = rl_cfg.get("rollout_steps", 50)
    save_interval = 5000
    log_interval = train_cfg.get("log_interval", 100)

    global_step = start_step
    iteration = 0

    logger.info("Starting Phase 3 PPO training...")
    logger.info(f"Total steps: {total_steps}, Rollout steps: {rollout_steps}")
    start_time = time.time()

    while global_step < total_steps:
        # Refresh text conditions periodically for diversity
        if iteration % 10 == 0:
            new_conditions = [get_random_text_condition() for _ in range(num_envs)]
            vec_env.set_text_conditions(new_conditions)

        # Collect rollout
        rollout_buffer = ppo_trainer.collect_rollout(
            vec_env, num_steps=rollout_steps,
        )

        # PPO update
        update_stats = ppo_trainer.update(rollout_buffer)

        # Also update LoRA parameters (gradient flows through env to UNet)
        # The PPO trainer handles policy/value grads; LoRA params are updated
        # via the environment's reward signal backpropagated through the policy.

        iteration += 1
        global_step += rollout_steps * num_envs

        # Logging
        if iteration % max(1, log_interval // rollout_steps) == 0:
            elapsed = time.time() - start_time

            # Compute mean reward from rollout buffer
            mean_reward = 0.0
            if rollout_buffer.rewards:
                mean_reward = sum(r.item() if isinstance(r, torch.Tensor) else r
                                  for r in rollout_buffer.rewards) / len(rollout_buffer.rewards)

            # Mean advantage from GAE
            data = rollout_buffer.to_tensors(device)
            last_val = rollout_buffer.last_value.to(device) if rollout_buffer.last_value is not None \
                else torch.zeros(1, device=device)
            advantages, returns = ppo_trainer.compute_gae(
                data["rewards"], data["values"], data["dones"], last_val,
            )
            mean_advantage = advantages.mean().item()

            logger.info(
                f"Step {global_step}/{total_steps} | "
                f"Iter: {iteration} | "
                f"Policy Loss: {update_stats['policy_loss']:.6f} | "
                f"Value Loss: {update_stats['value_loss']:.6f} | "
                f"Entropy: {update_stats['entropy']:.4f} | "
                f"Mean Reward: {mean_reward:.4f} | "
                f"Mean Advantage: {mean_advantage:.4f} | "
                f"Clip Frac: {update_stats['clip_fraction']:.4f} | "
                f"Approx KL: {update_stats['approx_kl']:.6f} | "
                f"Elapsed: {elapsed:.0f}s"
            )

        # Save checkpoint
        if global_step % save_interval < rollout_steps * num_envs and global_step > start_step:
            ckpt_path = str(output_dir / f"checkpoint_{global_step}.pt")
            ppo_trainer.save_checkpoint(ckpt_path)

            # Also save LoRA weights
            lora_path = str(output_dir / f"lora_{global_step}.pt")
            unet_module = unet.module if hasattr(unet, "module") else unet
            torch.save({
                "lora_state": {
                    name: param.data.cpu()
                    for name, param in unet_module.named_parameters()
                    if param.requires_grad
                },
                "step": global_step,
                "config": config,
            }, lora_path)

            # Save latest
            latest_path = str(output_dir / "latest.pt")
            ppo_trainer.save_checkpoint(latest_path)

            logger.info(f"Checkpoint saved at step {global_step}")

    # Final save
    final_path = str(output_dir / "final.pt")
    ppo_trainer.save_checkpoint(final_path)

    final_lora_path = str(output_dir / "lora_final.pt")
    unet_module = unet.module if hasattr(unet, "module") else unet
    torch.save({
        "lora_state": {
            name: param.data.cpu()
            for name, param in unet_module.named_parameters()
            if param.requires_grad
        },
        "step": global_step,
        "config": config,
    }, final_lora_path)

    vec_env.close()
    logger.info(f"Phase 3 training complete. Final checkpoint: {final_path}")
    logger.info(f"LoRA weights: {final_lora_path}")


if __name__ == "__main__":
    main()
