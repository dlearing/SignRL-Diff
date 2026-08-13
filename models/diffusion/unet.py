"""
signrl_diff.models.diffusion.unet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Video diffusion UNet for the SignRL-Diff pipeline.

The network denoises a latent video tensor ``z ∈ R^{B × T × C × H × W}``
conditioned on a diffusion timestep ``k`` and a text embedding
``c ∈ R^{B × L × D_text}``.  Each residual block is augmented with
**spatial self-attention**, **temporal self-attention**, and
**cross-attention** over the text condition.

LoRA adapters can be injected into every attention projection via
:py:meth:`VideoUNet.inject_lora`.

References
----------
* Ho et al., "Video Diffusion Models", 2022.
* Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2022.
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRALinear


# ======================================================================
# Positional / Timestep Embedding
# ======================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al.) followed by a
    two-layer MLP that produces a timestep conditioning vector.

    Parameters
    ----------
    dim : int
        Output dimensionality of the embedding.
    max_period : float, default 10000.0
        Controls the frequency range of the sinusoids.
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim: int = dim
        self.max_period: float = max_period

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed integer timesteps.

        Parameters
        ----------
        t : torch.Tensor
            1-D integer tensor of shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Embedding of shape ``(B, dim)``.
        """
        t_float = t.float().unsqueeze(-1)  # (B, 1)
        args = t_float * self.freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
        if self.dim % 2 == 1:
            # Pad with a zero column for odd dimensions
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ======================================================================
# Basic Layers
# ======================================================================

class AdaGroupNorm(nn.Module):
    """GroupNorm with adaptive scale/shift from a conditioning vector.

    Parameters
    ----------
    num_groups : int
        Number of groups for GroupNorm.
    num_channels : int
        Number of channels.
    cond_dim : int
        Dimensionality of the conditioning vector.
    """

    def __init__(
        self, num_groups: int, num_channels: int, cond_dim: int
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.scale = nn.Linear(cond_dim, num_channels)
        self.shift = nn.Linear(cond_dim, num_channels)
        # Initialise to identity
        nn.init.ones_(self.scale.weight.data.new_zeros(self.scale.weight.shape))
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Apply adaptive group normalisation.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, C, ...)`` input tensor.
        cond : torch.Tensor
            ``(B, cond_dim)`` conditioning vector.

        Returns
        -------
        torch.Tensor
            Normalised tensor, same shape as *x*.
        """
        h = self.norm(x)
        # cond → (B, C, 1, ...) for broadcasting
        scale = self.scale(cond)
        shift = self.shift(cond)
        # Reshape for spatial broadcasting
        for _ in range(h.dim() - 2):
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        return h * scale + shift


# ======================================================================
# Residual Block
# ======================================================================

class ResBlock(nn.Module):
    """Time-conditioned residual convolution block.

    Applies two 3×3 convolutions with adaptive group-norm and SiLU
    activations.  A 1×1 skip projection is used when the input and
    output channel counts differ.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    cond_dim : int
        Dimensionality of the timestep embedding vector.
    """

    def __init__(
        self, in_channels: int, out_channels: int, cond_dim: int
    ) -> None:
        super().__init__()
        self.norm1 = AdaGroupNorm(
            min(in_channels // 4, 32) or 1, in_channels, cond_dim
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaGroupNorm(
            min(out_channels // 4, 32) or 1, out_channels, cond_dim
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()

        # Skip projection
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_proj = nn.Identity()

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Apply the residual block.

        Parameters
        ----------
        x : torch.Tensor
            ``(B*T, C, H, W)`` input feature map.
        cond : torch.Tensor
            ``(B*T, cond_dim)`` timestep conditioning.

        Returns
        -------
        torch.Tensor
            ``(B*T, out_channels, H, W)`` output.
        """
        h = self.act(self.norm1(x, cond))
        h = self.conv1(h)
        h = self.act(self.norm2(h, cond))
        h = self.conv2(h)
        return h + self.skip_proj(x)


# ======================================================================
# Attention Modules
# ======================================================================

class SpatialAttention(nn.Module):
    """Self-attention over spatial dimensions, applied independently to
    each frame.

    Input tensor of shape ``(B, T, C, H, W)`` is reshaped so that
    attention is computed over the ``H×W`` tokens for each of the
    ``B×T`` frames.

    Parameters
    ----------
    channels : int
        Number of channels.
    num_heads : int, default 8
        Number of attention heads.
    """

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        self.channels: int = channels
        self.num_heads: int = num_heads
        self.head_dim: int = channels // num_heads

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial self-attention.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, T, C, H, W)`` feature tensor.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        B, T, C, H, W = x.shape
        # (B*T, H*W, C)
        h = x.permute(0, 1, 3, 4, 2).reshape(B * T, H * W, C)
        h = self.norm(h)

        qkv = self.qkv(h)  # (B*T, HW, 3C)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B*T, HW, C)

        # Multi-head reshape
        q = q.reshape(B * T, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B * T, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B * T, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B*T, heads, HW, head_dim)

        # Merge heads
        out = out.permute(0, 2, 1, 3).reshape(B * T, H * W, C)
        out = self.proj_out(out)

        # Residual
        out = out.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)
        return x + out


class TemporalAttention(nn.Module):
    """Self-attention over the temporal dimension, applied
    independently to each spatial position.

    Input ``(B, T, C, H, W)`` is rearranged so that attention is
    computed over ``T`` tokens for each of the ``B×H×W`` positions.

    Parameters
    ----------
    channels : int
        Number of channels.
    num_heads : int, default 8
        Number of attention heads.
    """

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        self.channels: int = channels
        self.num_heads: int = num_heads
        self.head_dim: int = channels // num_heads

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute temporal self-attention.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, T, C, H, W)`` feature tensor.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        B, T, C, H, W = x.shape
        # (B*H*W, T, C)
        h = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        h = self.norm(h)

        qkv = self.qkv(h)  # (B*H*W, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Multi-head
        q = q.reshape(B * H * W, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B * H * W, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B * H * W, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).reshape(B * H * W, T, C)
        out = self.proj_out(out)

        # Reshape back
        out = out.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        return x + out


class CrossAttention(nn.Module):
    """Cross-attention between video features (query) and a text
    condition (key/value).

    Parameters
    ----------
    channels : int
        Number of video feature channels.
    text_dim : int, default 1024
        Dimensionality of the text embeddings.
    num_heads : int, default 8
        Number of attention heads.
    """

    def __init__(
        self, channels: int, text_dim: int = 1024, num_heads: int = 8
    ) -> None:
        super().__init__()
        self.channels: int = channels
        self.text_dim: int = text_dim
        self.num_heads: int = num_heads
        self.head_dim: int = channels // num_heads

        self.norm_video = nn.LayerNorm(channels)
        self.norm_text = nn.LayerNorm(text_dim)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(text_dim, channels)
        self.v_proj = nn.Linear(text_dim, channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(
        self, x: torch.Tensor, text: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-attention.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, T, C, H, W)`` video feature tensor.
        text : torch.Tensor
            ``(B, L, D_text)`` text condition tensor.

        Returns
        -------
        torch.Tensor
            Same shape as *x*.
        """
        B, T, C, H, W = x.shape
        L = text.shape[1]

        # Video → (B*T, H*W, C)
        h = x.permute(0, 1, 3, 4, 2).reshape(B * T, H * W, C)
        h = self.norm_video(h)

        # Text → (B*T, L, D_text)  — repeat across T frames
        ctx = text.unsqueeze(1).expand(B, T, L, self.text_dim)
        ctx = ctx.reshape(B * T, L, self.text_dim)
        ctx = self.norm_text(ctx)

        # Projections
        q = self.q_proj(h)   # (B*T, HW, C)
        k = self.k_proj(ctx)  # (B*T, L, C)
        v = self.v_proj(ctx)  # (B*T, L, C)

        # Multi-head
        q = q.reshape(B * T, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B * T, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B * T, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.permute(0, 2, 1, 3).reshape(B * T, H * W, C)
        out = self.proj_out(out)

        out = out.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)
        return x + out


# ======================================================================
# Attention Block (Spatial + Temporal + Cross)
# ======================================================================

class AttentionBlock(nn.Module):
    """Composite attention block: Spatial → Temporal → Cross-Attention.

    Parameters
    ----------
    channels : int
        Number of feature channels.
    text_dim : int
        Dimensionality of text embeddings.
    num_heads : int
        Number of attention heads.
    """

    def __init__(
        self, channels: int, text_dim: int = 1024, num_heads: int = 8
    ) -> None:
        super().__init__()
        self.spatial_attn = SpatialAttention(channels, num_heads)
        self.temporal_attn = TemporalAttention(channels, num_heads)
        self.cross_attn = CrossAttention(channels, text_dim, num_heads)

    def forward(
        self, x: torch.Tensor, text: torch.Tensor
    ) -> torch.Tensor:
        """Apply spatial, temporal, and cross attention sequentially.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, T, C, H, W)`` feature tensor.
        text : torch.Tensor
            ``(B, L, D_text)`` text embeddings.

        Returns
        -------
        torch.Tensor
            Same shape as *x*.
        """
        x = self.spatial_attn(x)
        x = self.temporal_attn(x)
        x = self.cross_attn(x, text)
        return x


# ======================================================================
# Down / Up / Middle Blocks
# ======================================================================

class DownBlock(nn.Module):
    """Encoder block: ``ResBlock → AttentionBlock → Downsample``.

    Parameters
    ----------
    in_channels : int
        Input channel count.
    out_channels : int
        Output channel count.
    cond_dim : int
        Timestep conditioning dimensionality.
    text_dim : int
        Text embedding dimensionality.
    num_heads : int
        Number of attention heads.
    downsample : bool
        Whether to apply a 2× spatial downsample at the end.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        text_dim: int = 1024,
        num_heads: int = 8,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.res_block = ResBlock(in_channels, out_channels, cond_dim)
        self.attn = AttentionBlock(out_channels, text_dim, num_heads)
        if downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.downsample = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, T, C, H, W)`` feature tensor.
        t_emb : torch.Tensor
            ``(B, cond_dim)`` timestep embedding.
        text : torch.Tensor
            ``(B, L, D_text)`` text condition.

        Returns
        -------
        x : torch.Tensor
            Output after down-sampling.
        skip : torch.Tensor
            Pre-downsample output for the skip connection.
        """
        B, T, C, H, W = x.shape

        # ResBlock expects (B*T, C, H, W)
        x_flat = x.reshape(B * T, C, H, W)
        t_rep = t_emb.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        x_flat = self.res_block(x_flat, t_rep)
        x = x_flat.reshape(B, T, -1, H, W)

        # Attention operates on (B, T, C, H, W)
        x = self.attn(x, text)

        skip = x  # save for skip connection

        # Downsample spatial (per-frame)
        if not isinstance(self.downsample, nn.Identity):
            _, _, C_new, _, _ = x.shape
            x_flat = x.reshape(B * T, C_new, H, W)
            x_flat = self.downsample(x_flat)
            H_new, W_new = x_flat.shape[2], x_flat.shape[3]
            x = x_flat.reshape(B, T, C_new, H_new, W_new)

        return x, skip


class UpBlock(nn.Module):
    """Decoder block: skip-concat → ``ResBlock → AttentionBlock → Upsample``.

    Parameters
    ----------
    in_channels : int
        Input channel count (after skip concatenation).
    out_channels : int
        Output channel count.
    skip_channels : int
        Channel count of the skip tensor.
    cond_dim : int
        Timestep conditioning dimensionality.
    text_dim : int
        Text embedding dimensionality.
    num_heads : int
        Number of attention heads.
    upsample : bool
        Whether to apply a 2× spatial upsample.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        cond_dim: int,
        text_dim: int = 1024,
        num_heads: int = 8,
        upsample: bool = True,
    ) -> None:
        super().__init__()
        # After concat: in_channels + skip_channels
        self.res_block = ResBlock(
            in_channels + skip_channels, out_channels, cond_dim
        )
        self.attn = AttentionBlock(out_channels, text_dim, num_heads)
        if upsample:
            self.upsample_conv = nn.Conv2d(
                in_channels, in_channels, 3, padding=1
            )
        else:
            self.upsample_conv = nn.Identity()
        self.do_upsample: bool = upsample

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, T, C, H, W)`` feature tensor from the previous level.
        skip : torch.Tensor
            ``(B, T, C_skip, H_s, W_s)`` skip tensor from the encoder.
        t_emb : torch.Tensor
            ``(B, cond_dim)`` timestep embedding.
        text : torch.Tensor
            ``(B, L, D_text)`` text condition.

        Returns
        -------
        torch.Tensor
            ``(B, T, out_channels, H', W')`` output.
        """
        B, T, C, H, W = x.shape

        # Upsample spatial first (nearest-neighbor + conv)
        if self.do_upsample:
            x_flat = x.reshape(B * T, C, H, W)
            x_flat = F.interpolate(
                x_flat, scale_factor=2, mode="nearest"
            )
            x_flat = self.upsample_conv(x_flat)
            H_up, W_up = x_flat.shape[2], x_flat.shape[3]
            x = x_flat.reshape(B, T, C, H_up, W_up)

        # Handle potential spatial size mismatch with skip
        _, _, _, H_x, W_x = x.shape
        _, _, _, H_s, W_s = skip.shape
        if (H_x != H_s) or (W_x != W_s):
            x_flat = x.reshape(B * T, C, H_x, W_x)
            x_flat = F.interpolate(
                x_flat, size=(H_s, W_s), mode="bilinear", align_corners=False
            )
            x = x_flat.reshape(B, T, C, H_s, W_s)

        # Concatenate along channel dim
        x = torch.cat([x, skip], dim=2)  # (B, T, C+C_skip, H_s, W_s)

        _, _, C_cat, H_f, W_f = x.shape

        # ResBlock
        x_flat = x.reshape(B * T, C_cat, H_f, W_f)
        t_rep = t_emb.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        x_flat = self.res_block(x_flat, t_rep)
        x = x_flat.reshape(B, T, -1, H_f, W_f)

        # Attention
        x = self.attn(x, text)
        return x


class MiddleBlock(nn.Module):
    """Bottleneck block: ``ResBlock → AttentionBlock → ResBlock``.

    Parameters
    ----------
    channels : int
        Channel count (kept constant).
    cond_dim : int
        Timestep conditioning dimensionality.
    text_dim : int
        Text embedding dimensionality.
    num_heads : int
        Number of attention heads.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        text_dim: int = 1024,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.res_block1 = ResBlock(channels, channels, cond_dim)
        self.attn = AttentionBlock(channels, text_dim, num_heads)
        self.res_block2 = ResBlock(channels, channels, cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """Apply middle block.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, T, C, H, W)`` feature tensor.
        t_emb : torch.Tensor
            ``(B, cond_dim)`` timestep embedding.
        text : torch.Tensor
            ``(B, L, D_text)`` text condition.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        B, T, C, H, W = x.shape

        x_flat = x.reshape(B * T, C, H, W)
        t_rep = t_emb.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)

        x_flat = self.res_block1(x_flat, t_rep)
        x = x_flat.reshape(B, T, C, H, W)

        x = self.attn(x, text)

        x_flat = x.reshape(B * T, C, H, W)
        x_flat = self.res_block2(x_flat, t_rep)
        x = x_flat.reshape(B, T, C, H, W)
        return x


# ======================================================================
# VideoUNet
# ======================================================================

class VideoUNet(nn.Module):
    """Video diffusion UNet with spatial/temporal/cross-attention.

    Accepts a noisy latent ``z ∈ R^{B × T × 4 × H × W}``, a timestep
    tensor ``k ∈ Z^B``, and a text condition ``c ∈ R^{B × L × 1024}``,
    and predicts the noise ``ε̂`` of the same shape as *z*.

    Parameters
    ----------
    in_channels : int, default 4
        Number of input (and output) latent channels.
    channel_config : list[int], default [128, 256, 512, 512]
        Channel counts at each resolution level.
    text_dim : int, default 1024
        Dimensionality of the text condition embeddings.
    cond_dim : int, default 512
        Internal dimensionality of the timestep embedding.
    num_heads : int, default 8
        Number of attention heads.
    """

    def __init__(
        self,
        in_channels: int = 4,
        channel_config: Sequence[int] | None = None,
        text_dim: int = 1024,
        cond_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        if channel_config is None:
            channel_config = [128, 256, 512, 512]
        self.channel_config: List[int] = list(channel_config)
        self.text_dim: int = text_dim
        self.cond_dim: int = cond_dim
        n_levels: int = len(self.channel_config)

        # --- Timestep embedding ---
        self.time_embed = SinusoidalTimestepEmbedding(cond_dim)

        # --- Stem ---
        self.stem = nn.Conv2d(in_channels, self.channel_config[0], 3, padding=1)

        # --- Encoder (down) ---
        # Track output channels of each down block (= skip channels)
        down_out_channels: List[int] = []
        self.down_blocks = nn.ModuleList()
        for i in range(n_levels):
            c_in = self.channel_config[i]
            if i < n_levels - 1:
                c_out = self.channel_config[i + 1]
                do_down = True
            else:
                c_out = self.channel_config[i]
                do_down = False
            down_out_channels.append(c_out)
            self.down_blocks.append(
                DownBlock(
                    c_in, c_out, cond_dim, text_dim, num_heads,
                    downsample=do_down,
                )
            )

        # --- Bottleneck ---
        bottleneck_ch = self.channel_config[-1]
        self.middle_block = MiddleBlock(
            bottleneck_ch, cond_dim, text_dim, num_heads
        )

        # --- Decoder (up) ---
        # Skips are consumed in reverse order: skip[-1], skip[-2], ..., skip[0]
        reversed_skip_channels: List[int] = list(reversed(down_out_channels))

        # Build channel plan for each up block:
        #   up_in_ch[i]  = channels arriving from previous block (or bottleneck)
        #   up_out_ch[i] = channels after ResBlock (becomes input to next block)
        #   skip_ch[i]   = channels of the skip tensor to concatenate
        #   do_upsample  = False for the first block (i=0), True otherwise
        up_out_channels: List[int] = []
        for i in range(n_levels):
            if i < n_levels - 1:
                up_out_channels.append(self.channel_config[n_levels - 1 - i])
            else:
                up_out_channels.append(self.channel_config[0])

        self.up_blocks = nn.ModuleList()
        current_in_ch = bottleneck_ch
        for i in range(n_levels):
            skip_ch = reversed_skip_channels[i]
            out_ch = up_out_channels[i]
            do_up = (i > 0)  # first up block does NOT upsample
            self.up_blocks.append(
                UpBlock(
                    current_in_ch, out_ch, skip_ch,
                    cond_dim, text_dim, num_heads,
                    upsample=do_up,
                )
            )
            current_in_ch = out_ch

        # --- Output ---
        self.out_norm = nn.GroupNorm(
            min(self.channel_config[0] // 4, 32) or 1,
            self.channel_config[0],
        )
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(
            self.channel_config[0], in_channels, 3, padding=1
        )
        # Zero-init the final conv for stable training start
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
        timestep: torch.Tensor,
        text_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise component of the noisy latent.

        Parameters
        ----------
        z : torch.Tensor
            Noisy latent of shape ``(B, T, C, H, W)``.
        timestep : torch.Tensor
            Integer timestep tensor of shape ``(B,)``.
        text_condition : torch.Tensor
            Text embeddings of shape ``(B, L, D_text)``.

        Returns
        -------
        torch.Tensor
            Predicted noise ``ε̂``, same shape as *z*.
        """
        B, T, C, H, W = z.shape

        # --- Timestep embedding ---
        t_emb = self.time_embed(timestep)  # (B, cond_dim)

        # --- Stem (per-frame 2D conv) ---
        z_flat = z.reshape(B * T, C, H, W)
        h_flat = self.stem(z_flat)  # (B*T, ch0, H, W)
        ch0 = h_flat.shape[1]
        h = h_flat.reshape(B, T, ch0, H, W)

        # --- Encoder ---
        skips: List[torch.Tensor] = []
        for block in self.down_blocks:
            h, skip = block(h, t_emb, text_condition)
            skips.append(skip)

        # --- Bottleneck ---
        h = self.middle_block(h, t_emb, text_condition)

        # --- Decoder (reverse skip order) ---
        for i, block in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            h = block(h, skip, t_emb, text_condition)

        # --- Output head ---
        _, _, C_out, H_out, W_out = h.shape
        h_flat = h.reshape(B * T, C_out, H_out, W_out)
        h_flat = self.out_act(self.out_norm(h_flat))
        h_flat = self.out_conv(h_flat)  # (B*T, in_channels, H_out, W_out)
        noise_pred = h_flat.reshape(B, T, -1, H_out, W_out)

        return noise_pred

    # ------------------------------------------------------------------
    # LoRA helpers
    # ------------------------------------------------------------------

    def inject_lora(self, rank: int = 16, alpha: float = 16.0) -> None:
        """Replace every ``nn.Linear`` inside attention modules with
        :class:`LoRALinear`.

        Parameters
        ----------
        rank : int, default 16
            LoRA rank.
        alpha : float, default 16.0
            LoRA scaling factor.
        """
        # Walk all sub-modules and replace Linear layers that live
        # inside attention blocks (SpatialAttention, TemporalAttention,
        # CrossAttention).
        attn_types = (SpatialAttention, TemporalAttention, CrossAttention)
        for module in self.modules():
            if isinstance(module, attn_types):
                self._replace_linears_in_module(module, rank, alpha)

    @staticmethod
    def _replace_linears_in_module(
        module: nn.Module, rank: int, alpha: float
    ) -> None:
        """Recursively replace ``nn.Linear`` children of *module* with
        ``LoRALinear`` (skipping any that are already wrapped)."""
        for name, child in module.named_children():
            if isinstance(child, LoRALinear):
                # Already wrapped — skip
                continue
            if isinstance(child, nn.Linear):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, name, lora_layer)
            else:
                # Recurse into nested modules
                VideoUNet._replace_linears_in_module(child, rank, alpha)

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Return a flat list of all trainable LoRA parameters.

        Returns
        -------
        list[nn.Parameter]
            List of ``lora_A`` and ``lora_B`` parameters from every
            :class:`LoRALinear` in the model.
        """
        params: List[nn.Parameter] = []
        for module in self.modules():
            if isinstance(module, LoRALinear):
                params.append(module.lora_A)
                params.append(module.lora_B)
        return params

    def merge_lora(self) -> None:
        """Merge all LoRA adapters into their base weights."""
        for module in self.modules():
            if isinstance(module, LoRALinear) and not module.merged:
                module.merge()

    def reset_lora(self) -> None:
        """Reset all LoRA adapters to zero contribution."""
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.reset()
