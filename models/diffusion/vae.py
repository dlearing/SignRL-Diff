"""
signrl_diff.models.diffusion.vae
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simplified video Variational Auto-Encoder (VAE) with KL divergence for
the SignRL-Diff pipeline.

The encoder compresses a video tensor of shape ``(B, T, 3, 256, 256)``
into a latent representation of shape ``(B, T, 4, 32, 32)``, applying
2-D convolutions independently per frame (8× spatial downsampling over
three stages).  The decoder inverts this mapping.

Training uses the standard ELBO objective with the reparameterisation
trick and a closed-form KL divergence against a standard normal prior.

Reference
---------
Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Building blocks
# ======================================================================

class ResBlock2D(nn.Module):
    """Simple 2-D residual block with GroupNorm and SiLU.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups_in = min(in_channels // 4, 32) or 1
        groups_out = min(out_channels // 4, 32) or 1

        self.net = nn.Sequential(
            nn.GroupNorm(groups_in, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups_out, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.net(x) + self.skip(x)


class Downsample2D(nn.Module):
    """2× spatial down-sampling via strided convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    """2× spatial up-sampling via nearest-neighbour interpolation
    followed by a 3×3 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ======================================================================
# Encoder
# ======================================================================

class Encoder(nn.Module):
    """Video encoder: ``(B*T, 3, 256, 256) → (B*T, 2*latent_ch, 32, 32)``.

    Three downsampling stages (8× spatial reduction).

    Parameters
    ----------
    latent_channels : int, default 4
        Number of latent channels (the output will have ``2 * latent_channels``
        to parameterise both *mu* and *log-variance*).
    base_channels : int, default 64
        Channel count at the first resolution level.
    """

    def __init__(
        self, latent_channels: int = 4, base_channels: int = 64
    ) -> None:
        super().__init__()
        ch = base_channels

        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)

        # Level 1: ch → ch, then downsample to 2*ch
        self.block1 = nn.Sequential(
            ResBlock2D(ch, ch),
            ResBlock2D(ch, ch),
        )
        self.down1 = Downsample2D(ch)

        # Level 2: ch → 2*ch, then downsample to 4*ch
        self.block2 = nn.Sequential(
            ResBlock2D(ch, ch * 2),
            ResBlock2D(ch * 2, ch * 2),
        )
        self.down2 = Downsample2D(ch * 2)

        # Level 3: 2*ch → 4*ch, then downsample to 4*ch
        self.block3 = nn.Sequential(
            ResBlock2D(ch * 2, ch * 4),
            ResBlock2D(ch * 4, ch * 4),
        )
        self.down3 = Downsample2D(ch * 4)

        # Final projection to 2 * latent_channels
        self.norm_out = nn.GroupNorm(
            min(ch * 4 // 4, 32), ch * 4
        )
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(ch * 4, 2 * latent_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode frames into latent distribution parameters.

        Parameters
        ----------
        x : torch.Tensor
            ``(B*T, 3, H, W)`` frame tensor.

        Returns
        -------
        mu : torch.Tensor
            ``(B*T, latent_ch, H/8, W/8)`` mean.
        logvar : torch.Tensor
            ``(B*T, latent_ch, H/8, W/8)`` log-variance.
        """
        h = self.conv_in(x)
        h = self.block1(h)
        h = self.down1(h)
        h = self.block2(h)
        h = self.down2(h)
        h = self.block3(h)
        h = self.down3(h)
        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)

        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar


# ======================================================================
# Decoder
# ======================================================================

class Decoder(nn.Module):
    """Video decoder: ``(B*T, latent_ch, 32, 32) → (B*T, 3, 256, 256)``.

    Three upsampling stages (8× spatial increase), mirroring the encoder.

    Parameters
    ----------
    latent_channels : int, default 4
        Number of latent channels.
    base_channels : int, default 64
        Channel count at the first (highest) resolution level.
    """

    def __init__(
        self, latent_channels: int = 4, base_channels: int = 64
    ) -> None:
        super().__init__()
        ch = base_channels

        # Input projection
        self.conv_in = nn.Conv2d(latent_channels, ch * 4, 3, padding=1)

        # Level 1 (32×32): 4*ch → 4*ch, then upsample
        self.block1 = nn.Sequential(
            ResBlock2D(ch * 4, ch * 4),
            ResBlock2D(ch * 4, ch * 4),
        )
        self.up1 = Upsample2D(ch * 4)

        # Level 2 (64×64): 4*ch → 2*ch, then upsample
        self.block2 = nn.Sequential(
            ResBlock2D(ch * 4, ch * 2),
            ResBlock2D(ch * 2, ch * 2),
        )
        self.up2 = Upsample2D(ch * 2)

        # Level 3 (128×128): 2*ch → ch, then upsample
        self.block3 = nn.Sequential(
            ResBlock2D(ch * 2, ch),
            ResBlock2D(ch, ch),
        )
        self.up3 = Upsample2D(ch)

        # Final projection
        self.norm_out = nn.GroupNorm(min(ch // 4, 32) or 1, ch)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor into frames.

        Parameters
        ----------
        z : torch.Tensor
            ``(B*T, latent_ch, H/8, W/8)`` latent.

        Returns
        -------
        torch.Tensor
            ``(B*T, 3, H, W)`` reconstructed frames.
        """
        h = self.conv_in(z)
        h = self.block1(h)
        h = self.up1(h)
        h = self.block2(h)
        h = self.up2(h)
        h = self.block3(h)
        h = self.up3(h)
        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)
        return h


# ======================================================================
# AutoencoderKL
# ======================================================================

@dataclass
class VAELoss:
    """Container for the VAE training loss components.

    Attributes
    ----------
    total : torch.Tensor
        Total loss (reconstruction + KL).
    recon : torch.Tensor
        Reconstruction loss (L1 or L2).
    kl : torch.Tensor
        KL divergence against N(0, I).
    """

    total: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor


class AutoencoderKL(nn.Module):
    """Video VAE with KL-divergence regularisation.

    Maps a video tensor ``(B, T, 3, 256, 256)`` to a latent space of
    shape ``(B, T, 4, 32, 32)`` and back, using 2-D convolutions
    applied independently per frame.

    Parameters
    ----------
    latent_channels : int, default 4
        Number of channels in the latent space.
    base_channels : int, default 64
        Base channel count for the encoder/decoder.
    kl_weight : float, default 1e-4
        Scaling factor for the KL divergence term.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 64,
        kl_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        self.latent_channels: int = latent_channels
        self.kl_weight: float = kl_weight

        self.encoder = Encoder(latent_channels, base_channels)
        self.decoder = Decoder(latent_channels, base_channels)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(
        self, video: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a video into the latent space.

        Parameters
        ----------
        video : torch.Tensor
            ``(B, T, 3, H, W)`` input video.

        Returns
        -------
        z : torch.Tensor
            ``(B, T, latent_ch, H/8, W/8)`` sampled latent code
            (via the reparameterisation trick during training).
        mu : torch.Tensor
            Mean of the approximate posterior.
        logvar : torch.Tensor
            Log-variance of the approximate posterior.
        """
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)

        mu, logvar = self.encoder(frames)  # (B*T, latent_ch, H/8, W/8)

        # Reparameterisation trick
        z = self._reparameterize(mu, logvar)

        _, lc, Hl, Wl = z.shape
        z = z.reshape(B, T, lc, Hl, Wl)
        mu = mu.reshape(B, T, lc, Hl, Wl)
        logvar = logvar.reshape(B, T, lc, Hl, Wl)
        return z, mu, logvar

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor back into video space.

        Parameters
        ----------
        latent : torch.Tensor
            ``(B, T, latent_ch, H/8, W/8)`` latent code.

        Returns
        -------
        torch.Tensor
            ``(B, T, 3, H, W)`` reconstructed video.
        """
        B, T, lc, Hl, Wl = latent.shape
        z_flat = latent.reshape(B * T, lc, Hl, Wl)
        recon_flat = self.decoder(z_flat)  # (B*T, 3, H, W)
        _, C_out, H, W = recon_flat.shape
        return recon_flat.reshape(B, T, C_out, H, W)

    # ------------------------------------------------------------------
    # Forward (encode + decode, used during training)
    # ------------------------------------------------------------------

    def forward(
        self, video: torch.Tensor
    ) -> Tuple[torch.Tensor, VAELoss]:
        """Full forward pass: encode → sample → decode, compute loss.

        Parameters
        ----------
        video : torch.Tensor
            ``(B, T, 3, H, W)`` ground-truth video.

        Returns
        -------
        recon : torch.Tensor
            Reconstructed video, same shape as input.
        loss : VAELoss
            Composite training loss.
        """
        z, mu, logvar = self.encode(video)
        recon = self.decode(z)

        # Reconstruction loss (L1)
        recon_loss = F.l1_loss(recon, video)

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # Mean over the batch for stability
        kl_loss = self._kl_divergence(mu, logvar)

        total_loss = recon_loss + self.kl_weight * kl_loss

        return recon, VAELoss(total=total_loss, recon=recon_loss, kl=kl_loss)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample from ``N(mu, diag(exp(logvar)))`` via the
        reparameterisation trick.

        During evaluation (``model.eval()``), returns *mu* directly.

        Parameters
        ----------
        mu : torch.Tensor
            Mean tensor.
        logvar : torch.Tensor
            Log-variance tensor.

        Returns
        -------
        torch.Tensor
            Sampled latent code, same shape as *mu*.
        """
        if not torch.is_grad_enabled():
            # At inference time, skip stochasticity
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def _kl_divergence(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Closed-form KL divergence between ``N(mu, σ²)`` and ``N(0, 1)``.

        ``KL = -0.5 * Σ (1 + log(σ²) - μ² - σ²)``

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the approximate posterior.
        logvar : torch.Tensor
            Log-variance of the approximate posterior.

        Returns
        -------
        torch.Tensor
            Scalar KL loss (averaged over the batch dimension).
        """
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by batch size
        return kl / mu.shape[0]

    def get_latent_shape(self, video_shape: torch.Size) -> torch.Size:
        """Compute the latent tensor shape for a given video shape.

        Parameters
        ----------
        video_shape : torch.Size
            Shape of the input video ``(B, T, 3, H, W)``.

        Returns
        -------
        torch.Size
            Latent shape ``(B, T, latent_ch, H/8, W/8)``.
        """
        B, T, _, H, W = video_shape
        return torch.Size([B, T, self.latent_channels, H // 8, W // 8])
