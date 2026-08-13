"""
signrl_diff.models.diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Diffusion model components for the SignRL-Diff pipeline.

Public API
----------
- :class:`VideoUNet`    -- Video diffusion UNet with spatial/temporal/cross-attention.
- :class:`AutoencoderKL` -- Video VAE mapping pixels to/from latent space.
- :class:`LoRALinear`   -- Low-rank adapter wrapper for ``nn.Linear``.
- :class:`DDPMScheduler` -- DDPM noise schedule and sampling utilities.
"""

from .lora import LoRALinear
from .scheduler import DDPMScheduler
from .unet import VideoUNet
from .vae import AutoencoderKL

__all__ = [
    "VideoUNet",
    "AutoencoderKL",
    "LoRALinear",
    "DDPMScheduler",
]
