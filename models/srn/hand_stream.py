"""
Hand Articulation Stream for the Sub-Reward Network (SignRL-Diff).

Evaluates fine-grained hand pose quality from cropped hand regions
using a simplified EfficientNet-B4-style CNN backbone followed by
a temporal pooling and MLP head.

Input:  hand crops in R^{B, T, 2, 3, 128, 128}  (left + right hands, T frames)
Output: f_hand in R^{B, 128}
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MBConv (Mobile Inverted Bottleneck) block — core of EfficientNet
# ---------------------------------------------------------------------------

class _SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block.

    Args:
        channels: Number of input/output channels.
        reduction: Channel reduction ratio for the bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, squeezed, kernel_size=1)
        self.fc2 = nn.Conv2d(squeezed, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=[2, 3], keepdim=True)  # global avg pool
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class _MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution block (MBConv).

    This is the fundamental building block of EfficientNet.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        expand_ratio: Expansion factor for the hidden dimension.
        kernel_size: Depth-wise convolution kernel size.
        stride: Stride of the depth-wise convolution.
        se_reduction: Squeeze-excitation reduction ratio.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers: List[nn.Module] = []

        # Expansion phase (point-wise)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ])

        # Depth-wise convolution
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=kernel_size,
                stride=stride, padding=padding, groups=hidden_dim, bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        ])

        # Squeeze-and-Excitation
        layers.append(_SqueezeExcitation(hidden_dim, reduction=se_reduction))

        # Projection phase (point-wise, linear)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Simplified EfficientNet-B4 backbone
# ---------------------------------------------------------------------------

class _EfficientNetB4Backbone(nn.Module):
    """Simplified EfficientNet-B4 backbone producing 1792-dim features.

    The architecture follows the EfficientNet-B4 scaling (width x1.8,
    depth x1.4 compared to B0) but is implemented from scratch to avoid
    external weight dependencies.  It produces spatial feature maps that
    are globally pooled to a 1792-dimensional vector.

    Reference: Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*,
    ICML 2019.
    """

    def __init__(self) -> None:
        super().__init__()
        # Stem: 3 → 48 channels (B4 scales B0's 32 → 48)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )

        # MBConv stages — channel progression and block counts inspired by
        # EfficientNet-B4 scaling.
        # (in_ch, out_ch, num_blocks, expand, kernel, stride)
        stage_configs = [
            (48,  48,  2, 1, 3, 1),   # Stage 1
            (48,  80,  3, 6, 3, 2),   # Stage 2
            (80, 112,  3, 6, 5, 2),   # Stage 3
            (112, 192,  4, 6, 3, 2),   # Stage 4
            (192, 320,  4, 6, 5, 1),   # Stage 5
            (320, 576,  4, 6, 5, 2),   # Stage 6
            (576, 960,  5, 6, 3, 1),   # Stage 7
        ]

        stages: List[nn.Module] = []
        for in_ch, out_ch, num_blocks, expand, kernel, stride in stage_configs:
            blocks: List[nn.Module] = []
            for b in range(num_blocks):
                s = stride if b == 0 else 1
                ic = in_ch if b == 0 else out_ch
                blocks.append(
                    _MBConvBlock(ic, out_ch, expand_ratio=expand,
                                 kernel_size=kernel, stride=s)
                )
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        # Head: project to 1792 dims (matching EfficientNet-B4 output)
        self.head = nn.Sequential(
            nn.Conv2d(960, 1792, kernel_size=1, bias=False),
            nn.BatchNorm2d(1792),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape ``(N, 3, H, W)``.

        Returns:
            Feature tensor of shape ``(N, 1792)`` after global average pooling.
        """
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        # Global average pooling over spatial dims
        x = x.mean(dim=[2, 3])  # (N, 1792)
        return x


# ---------------------------------------------------------------------------
# Full Hand Articulation Stream
# ---------------------------------------------------------------------------

class HandArticulationStream(nn.Module):
    """EfficientNet-B4-style CNN for evaluating hand articulation quality.

    The stream processes cropped hand images from both left and right hands
    across all video frames.  Features are extracted by a simplified
    EfficientNet-B4 backbone (1792-d output), temporally and bilaterally
    pooled, then projected to a 128-d representation by a 2-layer MLP.

    Input pipeline:
        ``(B, T, 2, 3, 128, 128)`` → reshape → backbone → pool → MLP

    Args:
        feature_dim: Output dimensionality of the backbone.
        hidden_dim: Hidden layer size in the MLP head.
        output_dim: Final output dimensionality (f_hand).
        dropout: Dropout probability in the MLP.
    """

    def __init__(
        self,
        feature_dim: int = 1792,
        hidden_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = _EfficientNetB4Backbone()

        # 2-layer MLP head
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, hand_crops: torch.Tensor) -> torch.Tensor:
        """Forward pass of the hand articulation stream.

        Args:
            hand_crops: Cropped hand images of shape
                ``(B, T, 2, 3, 128, 128)`` where dim-2 indexes
                left/right hands.

        Returns:
            Hand feature vector ``f_hand`` of shape ``(B, 128)``.
        """
        B, T, H, C, Hh, Ww = hand_crops.shape  # H = 2 (left + right)

        # Merge batch, temporal, and hand dims for the backbone
        # (B, T, 2, 3, 128, 128) → (B*T*2, 3, 128, 128)
        x = hand_crops.reshape(B * T * H, C, Hh, Ww)

        # Extract features through backbone
        feats = self.backbone(x)  # (B*T*2, 1792)
        feat_dim = feats.shape[-1]

        # Reshape back: (B, T, 2, 1792)
        feats = feats.reshape(B, T, H, feat_dim)

        # Temporal average pooling over frames (dim=1) and hands (dim=2)
        feats = feats.mean(dim=2).mean(dim=1)  # (B, 1792)

        # MLP projection to output dim
        f_hand = self.mlp(feats)  # (B, 128)
        return f_hand
