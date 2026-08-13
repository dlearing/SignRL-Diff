"""
SignRL-Diff: RL-based Fine-Tuning of Video Diffusion Models
for Sign Language Generation
============================================================

This package implements the complete SignRL-Diff pipeline:

1. **Phase 1** — Pre-train a video diffusion model (UNet + VAE) on
   sign language datasets (PHOENIX14T, How2Sign, USTC-CSL).

2. **Phase 2** — Train a Sub-Reward Network (SRN) that evaluates
   generated videos along four complementary axes: body pose quality
   (GCN), temporal coherence (TCN), semantic alignment (CLIP cosine),
   and hand articulation (EfficientNet-B4).

3. **Phase 3** — Fine-tune the diffusion model via PPO with a
   Hierarchical Articulation-Aware Reward (HAR) that decomposes
   the terminal reward into intermediate per-step rewards.

Quick Start
-----------
::

    from signrl_diff.models import VideoUNet, AutoencoderKL, DDPMScheduler
    from signrl_diff.models import SubRewardNetwork
    from signrl_diff.models import PolicyNetwork, ValueNetwork, PPOTrainer
    from signrl_diff.rewards import HierarchicalReward
    from signrl_diff.env import DenoisingEnv, VecDenoisingEnv
    from signrl_diff.data import SignLanguageVideoDataset

Version
-------
"""

__version__ = "0.1.0"
__author__ = "SignRL-Diff Contributors"

from .models import (
    VideoUNet,
    AutoencoderKL,
    DDPMScheduler,
    SubRewardNetwork,
    PolicyNetwork,
    ValueNetwork,
    PPOTrainer,
)
from .rewards import HierarchicalReward
from .env import DenoisingEnv, VecDenoisingEnv
from .data import SignLanguageVideoDataset, build_preference_pairs
from .utils import (
    set_seed,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Version
    "__version__",
    # Models - Diffusion
    "VideoUNet",
    "AutoencoderKL",
    "DDPMScheduler",
    # Models - SRN
    "SubRewardNetwork",
    # Models - RL
    "PolicyNetwork",
    "ValueNetwork",
    "PPOTrainer",
    # Rewards
    "HierarchicalReward",
    # Environment
    "DenoisingEnv",
    "VecDenoisingEnv",
    # Data
    "SignLanguageVideoDataset",
    "build_preference_pairs",
    # Utils
    "set_seed",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
]
