"""
signrl_diff.models.rl.value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Value network for the SignRL-Diff PPO loop.

Estimates the state-value function ``V(s_k)`` used to compute
Generalised Advantage Estimation (GAE) targets.

The architecture mirrors :class:`PolicyNetwork`:

1. **State encoder** (shared design): flatten ``[z_k ; eps_hat_k]``, MLP, add PE(k).
2. **Cross-attention** over the text condition ``c``.
3. **Value head**: ``FC(512, 256) -> ReLU -> FC(256, 1)``.

References
----------
* Schulman et al., "High-Dimensional Continuous Control Using Generalized
  Advantage Estimation", 2016.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .policy import StateEncoder


class ValueNetwork(nn.Module):
    """Critic network that estimates the scalar state-value ``V(s_k)``.

    Uses the same :class:`StateEncoder` (state MLP + sinusoidal PE +
    cross-attention) as the policy, but replaces the action head with a
    two-layer MLP producing a single scalar.

    Parameters
    ----------
    num_frames : int
        Temporal length *T* of the latent video tensor.
    latent_channels : int
        Latent channel count (default 4).
    latent_hw : int
        Latent spatial height/width (default 32).
    text_dim : int
        Text embedding dimensionality (default 1024).
    hidden_dim : int
        Internal feature size (default 512).
    num_heads : int
        Cross-attention heads (default 8).
    """

    def __init__(
        self,
        num_frames: int = 16,
        latent_channels: int = 4,
        latent_hw: int = 32,
        text_dim: int = 1024,
        hidden_dim: int = 512,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        # Re-use the same encoder architecture as the policy
        self.encoder = StateEncoder(
            num_frames=num_frames,
            latent_channels=latent_channels,
            latent_hw=latent_hw,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )

        # Value head: FC(512, 256) -> ReLU -> FC(256, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        text_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the state-value estimate ``V(s_k)``.

        Parameters
        ----------
        state : tuple of ``(z_k, k, eps_hat_k)``
            - z_k : ``(B, T, C, H, W)`` noisy latent
            - k : timestep index (scalar or ``(B,)``)
            - eps_hat_k : ``(B, T, C, H, W)`` UNet noise prediction
        text_condition : Tensor, shape ``(B, L, D_text)``
            Text embedding.

        Returns
        -------
        value : Tensor, shape ``(B,)``
            Scalar state-value estimate.
        """
        z_k, k, eps_hat_k = state

        feat = self.encoder(z_k, k, text_condition, eps_hat_k)  # (B, hidden_dim)
        value = self.value_head(feat).squeeze(-1)                # (B,)

        return value
