"""
signrl_diff.models
~~~~~~~~~~~~~~~~~~

Model sub-packages for the SignRL-Diff pipeline.

Sub-packages
-------------
- ``diffusion``: Video UNet, VAE, DDPM scheduler, LoRA adapters.
- ``srn``: Sub-Reward Network (GCN, TCN, Semantic, Hand streams + fusion).
- ``rl``: Policy network, value network, PPO trainer.
"""

from .diffusion import VideoUNet, AutoencoderKL, LoRALinear, DDPMScheduler
from .srn import (
    SubRewardNetwork,
    PoseGCNStream,
    TemporalCoherenceStream,
    SemanticAlignmentStream,
    HandArticulationStream,
)
from .rl import PolicyNetwork, ValueNetwork, PPOTrainer

__all__ = [
    # Diffusion
    "VideoUNet",
    "AutoencoderKL",
    "LoRALinear",
    "DDPMScheduler",
    # SRN
    "SubRewardNetwork",
    "PoseGCNStream",
    "TemporalCoherenceStream",
    "SemanticAlignmentStream",
    "HandArticulationStream",
    # RL
    "PolicyNetwork",
    "ValueNetwork",
    "PPOTrainer",
]
