"""
signrl_diff.models.rl.policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lightweight transformer policy network for the SignRL-Diff RL loop.

The policy ``pi(a_k | s_k)`` maps a diffusion state

    s_k = (z_k, k, c, eps_hat_k)

to a 129-dimensional action

    a_k = [global(64) | hand_left(32) | hand_right(32) | scale(1)]

Architecture overview
---------------------
1. **Sinusoidal timestep embedding** PE(k) in R^{512}.
2. **State encoder**: flatten [z_k ; eps_hat_k], two-layer MLP to 512-d, add PE(k).
3. **Cross-attention**: single query from state, keys/values from text condition c.
4. **Action head**: MLP producing mean and log-std of a diagonal Gaussian.

References
----------
* Vaswani et al., "Attention Is All You Need", 2017.
* Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ======================================================================
# Timestep Embedding
# ======================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding for the diffusion timestep.

    For dimension index *i* (0-based, half-indexed) and timestep *k*:

        PE(k, 2i)   = sin(k / 10000^{2i/d})
        PE(k, 2i+1) = cos(k / 10000^{2i/d})

    Parameters
    ----------
    dim : int
        Total embedding dimensionality (must be even).
    max_period : float
        Controls the minimum frequency of the sinusoids.
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, f"Embedding dim must be even, got {dim}"
        self.dim: int = dim
        self.max_period: float = max_period

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32)
            / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embedding for timestep(s) *k*.

        Parameters
        ----------
        k : Tensor
            Scalar or 1-D tensor of timestep indices (integer-valued).

        Returns
        -------
        Tensor
            Shape ``(*k.shape, dim)``.
        """
        k_float = k.float()
        if k_float.dim() == 0:
            k_float = k_float.unsqueeze(0)

        # outer product: (..., half)
        args = k_float.unsqueeze(-1) * self.freqs  # (N, half)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (N, dim)

        if k.dim() == 0:
            emb = emb.squeeze(0)

        return emb


# ======================================================================
# Shared State Encoder (re-used by ValueNetwork)
# ======================================================================

class StateEncoder(nn.Module):
    """Encodes the full diffusion state into a 512-d feature vector
    conditioned on the text embedding via cross-attention.

    Pipeline
    --------
    1. Concatenate and flatten ``[z_k ; eps_hat_k]``.
    2. MLP: ``Linear(flat_dim, 1024) -> ReLU -> Linear(1024, 512)``.
    3. Add sinusoidal timestep embedding PE(k).
    4. Cross-attention: query = state feat (1 x 512), keys/values from text.

    Parameters
    ----------
    num_frames : int
        Temporal length *T* of the latent video tensor.
    latent_channels : int
        Channel count *C* of the latent (default 4).
    latent_hw : int
        Spatial height/width of the latent (default 32).
    text_dim : int
        Dimensionality of the text condition vectors (default 1024).
    hidden_dim : int
        Internal feature dimensionality (default 512).
    num_heads : int
        Number of attention heads for cross-attention (default 8).
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

        self.num_frames = num_frames
        self.latent_channels = latent_channels
        self.latent_hw = latent_hw

        # Flatten dimensionality for [z_k ; eps_hat_k]
        flat_dim = 2 * num_frames * latent_channels * latent_hw * latent_hw

        # --- State MLP ---
        self.state_mlp = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, hidden_dim),
        )

        # --- Timestep embedding ---
        self.timestep_emb = SinusoidalTimestepEmbedding(dim=hidden_dim)

        # --- Cross-attention projections ---
        self.text_proj_k = nn.Linear(text_dim, hidden_dim)
        self.text_proj_v = nn.Linear(text_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.hidden_dim = hidden_dim

    def forward(
        self,
        z_k: torch.Tensor,
        k: torch.Tensor,
        text_condition: torch.Tensor,
        eps_hat_k: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the full state and attend to the text condition.

        Parameters
        ----------
        z_k : Tensor, shape ``(B, T, C, H, W)``
            Noisy latent at timestep *k*.
        k : Tensor
            Timestep index, scalar or shape ``(B,)``.
        text_condition : Tensor, shape ``(B, L, D_text)``
            Text embedding from the language model.
        eps_hat_k : Tensor, shape ``(B, T, C, H, W)``
            UNet noise prediction at timestep *k*.

        Returns
        -------
        Tensor, shape ``(B, hidden_dim)``
            Fused state feature after cross-attention.
        """
        batch_size = z_k.shape[0]

        # (i) Flatten and concatenate noisy latent with predicted noise
        z_flat = z_k.reshape(batch_size, -1)           # (B, T*C*H*W)
        eps_flat = eps_hat_k.reshape(batch_size, -1)    # (B, T*C*H*W)
        state_flat = torch.cat([z_flat, eps_flat], dim=-1)  # (B, 2*T*C*H*W)

        # (ii) State MLP + timestep embedding
        state_feat = self.state_mlp(state_flat)          # (B, hidden_dim)

        # Timestep embedding — handle both scalar and batched k
        if k.dim() == 0:
            pe = self.timestep_emb(k)                    # (hidden_dim,)
            state_feat = state_feat + pe.unsqueeze(0)
        else:
            pe = self.timestep_emb(k)                    # (B, hidden_dim)
            state_feat = state_feat + pe

        # (iii) Cross-attention: Q = state, K/V = text
        query = state_feat.unsqueeze(1)                  # (B, 1, hidden_dim)
        keys = self.text_proj_k(text_condition)          # (B, L, hidden_dim)
        vals = self.text_proj_v(text_condition)          # (B, L, hidden_dim)

        attn_out, _ = self.cross_attn(query, keys, vals) # (B, 1, hidden_dim)
        out = attn_out.squeeze(1)                        # (B, hidden_dim)

        return out


# ======================================================================
# Policy Network
# ======================================================================

# Action dimensionality constants
GLOBAL_DIM: int = 64
HAND_LEFT_DIM: int = 32
HAND_RIGHT_DIM: int = 32
SCALE_DIM: int = 1
ACTION_DIM: int = GLOBAL_DIM + HAND_LEFT_DIM + HAND_RIGHT_DIM + SCALE_DIM  # 129


class PolicyNetwork(nn.Module):
    """Lightweight transformer policy for diffusion-guided RL.

    Outputs a diagonal-Gaussian distribution over 129-dimensional actions
    decomposed as::

        a_k = [global(64), hand_left(32), hand_right(32), scale(1)]

    The mean and log-standard-deviation are produced by a shared action
    head; actions are sampled as ``a ~ N(mu, diag(exp(2 * log_sigma)))``.

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
    action_dim : int
        Total action dimensionality (default 129).
    log_std_init : float
        Initial value for the learnable log-std bias (default -1.0).
    """

    def __init__(
        self,
        num_frames: int = 16,
        latent_channels: int = 4,
        latent_hw: int = 32,
        text_dim: int = 1024,
        hidden_dim: int = 512,
        num_heads: int = 8,
        action_dim: int = ACTION_DIM,
        log_std_init: float = -1.0,
    ) -> None:
        super().__init__()

        self.action_dim: int = action_dim

        # Shared state encoder + cross-attention
        self.encoder = StateEncoder(
            num_frames=num_frames,
            latent_channels=latent_channels,
            latent_hw=latent_hw,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )

        # Action head: FC(512,256) -> ReLU -> FC(256,128) -> ReLU -> FC(128, 2*action_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * action_dim),
        )

        # Initialise the log-std portion of the last layer to a small
        # negative value so the initial policy is near-deterministic.
        with torch.no_grad():
            self.action_head[-1].bias[action_dim:] = log_std_init

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_distribution(
        self,
        z_k: torch.Tensor,
        k: torch.Tensor,
        text_condition: torch.Tensor,
        eps_hat_k: torch.Tensor,
    ) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        """Build the action distribution from raw state inputs.

        Returns
        -------
        dist : Normal
            Diagonal Gaussian over actions.
        mean : Tensor, shape ``(B, action_dim)``
        log_std : Tensor, shape ``(B, action_dim)``
        """
        feat = self.encoder(z_k, k, text_condition, eps_hat_k)  # (B, hidden)
        head_out = self.action_head(feat)                        # (B, 2*action_dim)

        mean = head_out[:, : self.action_dim]
        log_std = head_out[:, self.action_dim :]

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)

        std = torch.exp(log_std)
        dist = Normal(mean, std)

        return dist, mean, log_std

    @staticmethod
    def _unpack_state(
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpack a state tuple ``(z_k, k, eps_hat_k)``."""
        z_k, k, eps_hat_k = state
        return z_k, k, eps_hat_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        text_condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log-probability.

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
        action : Tensor, shape ``(B, 129)``
            Sampled action.
        log_prob : Tensor, shape ``(B,)``
            Log-probability of the sampled action under the current policy.
        """
        z_k, k, eps_hat_k = self._unpack_state(state)
        dist, mean, log_std = self._get_distribution(
            z_k, k, text_condition, eps_hat_k
        )

        # Reparameterised sample
        action = dist.rsample()                                 # (B, action_dim)
        log_prob = dist.log_prob(action).sum(dim=-1)            # (B,)

        return action, log_prob

    def evaluate(
        self,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        text_condition: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the log-probability (and entropy) of stored actions
        under the **current** policy.  Used during PPO updates to compute
        the importance-sampling ratio.

        Parameters
        ----------
        state : tuple of ``(z_k, k, eps_hat_k)``
            Same format as :meth:`forward`.
        text_condition : Tensor, shape ``(B, L, D_text)``
        action : Tensor, shape ``(B, 129)``
            Previously sampled actions stored in the rollout buffer.

        Returns
        -------
        log_prob : Tensor, shape ``(B,)``
            Log-probability of *action* under the current policy.
        entropy : Tensor, shape ``(B,)``
            Entropy of the current policy.
        mean : Tensor, shape ``(B, 129)``
            Current action mean (useful for logging).
        """
        z_k, k, eps_hat_k = self._unpack_state(state)
        dist, mean, log_std = self._get_distribution(
            z_k, k, text_condition, eps_hat_k
        )

        log_prob = dist.log_prob(action).sum(dim=-1)   # (B,)
        entropy = dist.entropy().sum(dim=-1)            # (B,)

        return log_prob, entropy, mean

    def decompose_action(self, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split a 129-d action tensor into semantic components.

        Parameters
        ----------
        action : Tensor, shape ``(..., 129)``

        Returns
        -------
        dict with keys ``"global"``, ``"hand_left"``, ``"hand_right"``, ``"scale"``.
        """
        return {
            "global": action[..., :GLOBAL_DIM],
            "hand_left": action[..., GLOBAL_DIM : GLOBAL_DIM + HAND_LEFT_DIM],
            "hand_right": action[
                ...,
                GLOBAL_DIM + HAND_LEFT_DIM : GLOBAL_DIM + HAND_LEFT_DIM + HAND_RIGHT_DIM,
            ],
            "scale": action[..., -SCALE_DIM:],
        }
