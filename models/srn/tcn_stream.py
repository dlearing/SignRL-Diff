"""
Temporal Coherence Stream for the Sub-Reward Network (SignRL-Diff).

Evaluates temporal smoothness of generated sign-language videos via a
dilated causal 1-D convolution network operating on frame-to-frame pose
differences.

Input:  delta_J in R^{B, (T-1), 148, 3}  (flattened to R^{B, (T-1), 444})
Output: f_temp in R^{B, 256}
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dilated Causal Convolution Block
# ---------------------------------------------------------------------------

class DilatedCausalConvBlock(nn.Module):
    """Single dilated causal convolution block with residual connection.

    Each block applies:
        1. 1-D dilated convolution (causal via left-padding)
        2. Batch normalisation
        3. ReLU activation
        4. Residual addition (with 1x1 projection when channel sizes differ)

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel width.
        dilation: Dilation factor.
        dropout: Dropout probability applied after activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Left-padding for causal convolution: pad = (kernel_size - 1) * dilation
        self.causal_pad = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # manual causal padding applied in forward()
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=dropout)

        # 1x1 projection when residual channel sizes differ
        self.residual_proj: nn.Module | None = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape ``(B, C_in, L)`` where *L* is the
               temporal dimension.

        Returns:
            Output tensor of shape ``(B, C_out, L)`` (length preserved by
            causal padding).
        """
        # Causal left-padding: pad only the left side so that output length
        # equals input length (no future information leakage).
        padded = F.pad(x, (self.causal_pad, 0))  # (B, C_in, L + causal_pad)
        out = self.conv(padded)                   # (B, C_out, L)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Residual connection
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        out = out + residual
        return out


# ---------------------------------------------------------------------------
# Full Temporal Coherence Stream
# ---------------------------------------------------------------------------

class TemporalCoherenceStream(nn.Module):
    """1-D Temporal Convolutional Network for motion smoothness evaluation.

    The network processes frame-to-frame keypoint differences
    ``delta_J_t = J_{t+1} - J_t`` through 6 dilated causal convolution
    layers with exponentially increasing dilation rates
    ``d_l = 2^l`` for ``l in {0, 1, 2, 3, 4, 5}``.

    Receptive field:
        sum_{l=0}^{5} (kernel_size - 1) * 2^l = 2 * (1+2+4+8+16+32) = 126 frames

    Architecture per layer:
        Dilated Causal Conv1D → BatchNorm → ReLU → Dropout → Residual Add

    Final global average pooling over the temporal dimension produces
    ``f_temp in R^{256}``.

    Args:
        input_dim: Flattened per-frame feature dimension (148 joints * 3
            coords = 444 by default).
        hidden_channels: Number of channels inside each conv block.
        num_layers: Number of dilated causal conv layers.
        kernel_size: Convolution kernel width.
        dropout: Dropout probability within each block.
    """

    def __init__(
        self,
        input_dim: int = 444,
        hidden_channels: int = 256,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Project input to hidden_channels if necessary
        self.input_proj: nn.Module | None = None
        if input_dim != hidden_channels:
            self.input_proj = nn.Conv1d(input_dim, hidden_channels, kernel_size=1)

        # Build dilated causal conv blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            dilation = 2 ** layer_idx  # 1, 2, 4, 8, 16, 32
            in_ch = hidden_channels
            self.blocks.append(
                DilatedCausalConvBlock(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

    @property
    def receptive_field(self) -> int:
        """Theoretical receptive field in number of input frames."""
        kernel_size = 3
        total = 0
        for layer_idx in range(len(self.blocks)):
            total += (kernel_size - 1) * (2 ** layer_idx)
        return total

    def forward(self, pose_diff: torch.Tensor) -> torch.Tensor:
        """Forward pass of the temporal coherence stream.

        Args:
            pose_diff: Frame-to-frame keypoint differences of shape
                ``(B, T-1, 148, 3)``.  The caller may also pass the
                already-flattened variant ``(B, T-1, 444)``.

        Returns:
            Temporal feature vector ``f_temp`` of shape ``(B, 256)``.
        """
        # Flatten joint dims if needed: (B, T-1, 148, 3) → (B, T-1, 444)
        if pose_diff.dim() == 4:
            B, Tm1, V, C = pose_diff.shape
            pose_diff = pose_diff.reshape(B, Tm1, V * C)

        # Conv1d expects (B, C, L) — swap temporal and feature dims
        x = pose_diff.permute(0, 2, 1).contiguous()  # (B, 444, T-1)

        # Project to hidden_channels
        if self.input_proj is not None:
            x = self.input_proj(x)  # (B, 256, T-1)

        # Pass through dilated causal conv blocks
        for block in self.blocks:
            x = block(x)  # (B, 256, T-1)

        # Global average pooling over the temporal dimension
        f_temp = x.mean(dim=2)  # (B, 256)
        return f_temp
