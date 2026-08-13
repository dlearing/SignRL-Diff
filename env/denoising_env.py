"""
signrl_diff.env.denoising_env
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Denoising MDP environment for RL-based fine-tuning of the video diffusion
model in the SignRL-Diff pipeline.

The environment wraps one step of the reverse diffusion process as an MDP:

- **State**: ``s_k = (z_k, k, c, eps_hat_k)``
    - ``z_k``: noisy latent of shape ``(T, C, H, W)``
    - ``k``: integer diffusion timestep
    - ``c``: text condition of shape ``(L, D_text)``
    - ``eps_hat_k``: UNet noise prediction ``(T, C, H, W)``

- **Action**: ``a_k in R^129 = [global(64), hand_left(32), hand_right(32), scale(1)]``
    - ``global``: projected to full latent correction via learned ``W_global``
    - ``hand_left/right``: projected to hand-region correction via sparse ``M_hand``
    - ``scale``: modulates classifier-free guidance scale (+/- 2.0 around 7.5)

- **Transition**:
    1. ``eps_corr = W_global @ a_global + M_hand_left @ a_hand_l + M_hand_right @ a_hand_r``
    2. ``eps_tilde_k = eps_hat_k + eps_corr``
    3. ``z_{k-1} = scheduler.step(eps_tilde_k, k, z_k)``
    4. Compute reward (intermediate from HAR, or terminal if k=0)
    5. ``eps_hat_{k-1} = UNet(z_{k-1}, k-1, c)``
    6. Return ``(next_state, reward, done, info)``

The environment satisfies the :class:`~signrl_diff.models.rl.ppo.DiffusionEnv`
protocol expected by the PPO trainer.

References
----------
* SignRL-Diff: RL-based fine-tuning for sign language video generation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.helpers import action_to_correction, build_hand_sparse_projection


class DenoisingEnv:
    """Single denoising MDP environment.

    Parameters
    ----------
    unet : nn.Module
        Frozen video diffusion UNet (predicts noise).
    vae : nn.Module
        Video VAE (provides encoder/decoder).
    scheduler : nn.Module
        DDPM noise scheduler with ``step()`` method.
    reward_fn : nn.Module
        Reward function (e.g., :class:`HierarchicalReward`).
    text_condition : Tensor, shape ``(L, D_text)``
        Text embedding for this episode (fixed throughout the episode).
    K : int
        Total number of diffusion steps (default 50).
    num_frames : int
        Temporal length *T* of the latent video.
    latent_channels : int
        Latent channel count *C*.
    latent_hw : int
        Latent spatial height/width.
    cfg_scale_default : float
        Default classifier-free guidance scale.
    cfg_scale_range : float
        Modulation range for CFG scale (+/- around default).
    device : str or torch.device
        Target device.
    """

    def __init__(
        self,
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
        device: str | torch.device = "cpu",
    ) -> None:
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.reward_fn = reward_fn

        self.K: int = K
        self.num_frames: int = num_frames
        self.latent_channels: int = latent_channels
        self.latent_hw: int = latent_hw
        self.cfg_scale_default: float = cfg_scale_default
        self.cfg_scale_range: float = cfg_scale_range
        self.device = torch.device(device)

        # Store text condition (fixed per episode)
        self.text_condition: torch.Tensor = text_condition.to(self.device)

        # Latent flat dimensionality
        self.latent_flat = num_frames * latent_channels * latent_hw * latent_hw

        # ------------------------------------------------------------------
        # Action projection layers
        # ------------------------------------------------------------------
        # Learned linear projection: R^64 -> R^{latent_flat}
        self.w_global = nn.Linear(64, self.latent_flat, bias=False).to(self.device)
        # Initialize with small values to start with near-zero corrections
        nn.init.normal_(self.w_global.weight, mean=0.0, std=0.01)

        # Sparse hand projection matrices (fixed, not learned)
        self.m_hand_left = build_hand_sparse_projection(
            num_frames=num_frames,
            latent_channels=latent_channels,
            latent_hw=latent_hw,
            hand_action_dim=32,
            hand_region_ratio=0.25,
            device=self.device,
        )
        # Mirror for right hand (bottom-left quarter)
        self.m_hand_right = build_hand_sparse_projection(
            num_frames=num_frames,
            latent_channels=latent_channels,
            latent_hw=latent_hw,
            hand_action_dim=32,
            hand_region_ratio=0.25,
            device=self.device,
        )
        # Shift right-hand region to bottom-left quadrant
        # (the default builds bottom-right; we re-build for left)
        self._mirror_hand_projection(self.m_hand_right)

        # ------------------------------------------------------------------
        # Episode state
        # ------------------------------------------------------------------
        self._z_k: Optional[torch.Tensor] = None
        self._k: int = K
        self._eps_hat_k: Optional[torch.Tensor] = None
        self._done: bool = False

    def _mirror_hand_projection(self, m_hand: torch.Tensor) -> None:
        """Shift the hand projection from bottom-right to bottom-left spatial quadrant."""
        hw = self.latent_hw
        hand_hw = int(hw * 0.25 ** 0.5)
        hand_hw = max(hand_hw, 1)

        # Clear and rebuild for left side (w_start = 0)
        m_hand.zero_()
        region_size = self.num_frames * self.latent_channels * hand_hw * hand_hw
        for idx in range(min(32, region_size)):
            flat_in_region = idx % region_size
            t = flat_in_region // (self.latent_channels * hand_hw * hand_hw)
            rem = flat_in_region % (self.latent_channels * hand_hw * hand_hw)
            c = rem // (hand_hw * hand_hw)
            rem2 = rem % (hand_hw * hand_hw)
            h_local = rem2 // hand_hw
            w_local = rem2 % hand_hw

            h_abs = hw - hand_hw + h_local
            w_abs = w_local  # left side: starts at 0

            full_flat = (
                t * (self.latent_channels * hw * hw)
                + c * (hw * hw)
                + h_abs * hw
                + w_abs
            )
            if full_flat < self.latent_flat:
                m_hand[full_flat, idx] = 1.0 / max(32, 1)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self) -> Dict[str, Any]:
        """Build the observation dictionary from the current state.

        Returns
        -------
        dict with keys ``z_k``, ``k``, ``text_condition``, ``eps_hat_k``.
        """
        return {
            "z_k": self._z_k.detach().cpu(),
            "k": torch.tensor(self._k, dtype=torch.long),
            "text_condition": self.text_condition.detach().cpu(),
            "eps_hat_k": self._eps_hat_k.detach().cpu(),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        text_condition: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to the start of a denoising episode.

        Samples ``z_K ~ N(0, I)`` and computes the initial UNet
        prediction ``eps_hat_K``.

        Parameters
        ----------
        text_condition : Tensor, optional
            New text condition for this episode. If *None*, reuses the
            condition from ``__init__``.

        Returns
        -------
        obs : dict
            Initial observation.
        info : dict
            Auxiliary information (empty).
        """
        if text_condition is not None:
            self.text_condition = text_condition.to(self.device)

        # Sample pure noise
        self._z_k = torch.randn(
            1, self.num_frames, self.latent_channels,
            self.latent_hw, self.latent_hw,
            device=self.device,
        )
        self._k = self.K
        self._done = False

        # Initial UNet prediction (no gradient)
        with torch.no_grad():
            k_tensor = torch.full((1,), self._k, device=self.device, dtype=torch.long)
            text_batch = self.text_condition.unsqueeze(0)  # (1, L, D)
            self._eps_hat_k = self.unet(self._z_k, k_tensor, text_batch)

        return self._build_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self, action: torch.Tensor
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one denoising step.

        Parameters
        ----------
        action : Tensor, shape ``(129,)`` or ``(1, 129)``
            Action tensor ``[global(64), hand_left(32), hand_right(32), scale(1)]``.

        Returns
        -------
        obs : dict
            Next observation.
        reward : float
            Scalar reward.
        terminated : bool
            Whether the episode terminated normally (k reached 0).
        truncated : bool
            Whether the episode was truncated (always False for diffusion).
        info : dict
            Auxiliary information.
        """
        if self._done:
            return self._build_obs(), 0.0, True, False, {"already_done": True}

        # Ensure batch dim
        if action.dim() == 1:
            action = action.unsqueeze(0)  # (1, 129)
        action = action.to(self.device)

        # Decompose action
        a_global = action[:, :64]
        a_hand_left = action[:, 64:96]
        a_hand_right = action[:, 96:128]
        a_scale = action[:, 128:129]  # (1, 1)

        # Project action to latent correction
        eps_corr = action_to_correction(
            action,
            self.w_global,
            self.m_hand_left,
            self.m_hand_right,
            num_frames=self.num_frames,
            latent_channels=self.latent_channels,
            latent_hw=self.latent_hw,
        )  # (1, T, C, H, W)

        # Corrected noise prediction
        eps_tilde_k = self._eps_hat_k + eps_corr

        # CFG scale modulation
        cfg_scale = self.cfg_scale_default + a_scale.squeeze(-1) * self.cfg_scale_range
        cfg_scale = cfg_scale.clamp(
            min=self.cfg_scale_default - self.cfg_scale_range,
            max=self.cfg_scale_default + self.cfg_scale_range,
        )

        # Apply CFG: scale the noise prediction relative to unconditional
        # (simplified: we modulate the correction magnitude)
        cfg_factor = cfg_scale / self.cfg_scale_default
        eps_tilde_k = eps_tilde_k * cfg_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Scheduler reverse step: z_{k-1} = step(eps_tilde_k, k, z_k)
        z_next = self.scheduler.step(eps_tilde_k, self._k, self._z_k)

        # Compute reward
        info: Dict[str, Any] = {
            "step": self._k,
            "cfg_scale": cfg_scale.item(),
        }

        if self._k <= 1:
            # Terminal step: compute terminal reward
            reward = self.reward_fn.compute_terminal_reward(
                z_next, self.text_condition.unsqueeze(0)
            ).item()
            self._done = True
            terminated = True
            info["terminal"] = True

            # Set terminal state (clean latent, k=0)
            self._z_k = z_next
            self._k = 0
            self._eps_hat_k = torch.zeros_like(z_next)
        else:
            # Intermediate step: compute HAR intermediate reward
            reward = self.reward_fn.compute_intermediate_reward(
                z_next, self.text_condition.unsqueeze(0),
                self._k, self.K
            ).item()
            terminated = False

            # Advance state
            self._z_k = z_next
            self._k = self._k - 1

            # New UNet prediction for the next state
            k_tensor = torch.full((1,), self._k, device=self.device, dtype=torch.long)
            text_batch = self.text_condition.unsqueeze(0)
            self._eps_hat_k = self.unet(self._z_k, k_tensor, text_batch)

        return self._build_obs(), reward, terminated, False, info


class VecDenoisingEnv:
    """Vectorized wrapper running *N* parallel :class:`DenoisingEnv` instances.

    This provides batched observations and actions for efficient PPO
    rollout collection, satisfying the
    :class:`~signrl_diff.models.rl.ppo.VecDiffusionEnv` protocol.

    Parameters
    ----------
    env_fns : list of callable
        Each callable returns a :class:`DenoisingEnv` when called with
        no arguments.  Length determines ``num_envs``.
    """

    def __init__(self, env_fns: List[callable]) -> None:
        self.envs: List[DenoisingEnv] = [fn() for fn in env_fns]
        self.num_envs: int = len(self.envs)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        text_conditions: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset all environments and return stacked observations.

        Parameters
        ----------
        text_conditions : list of Tensor, optional
            Per-environment text conditions.  If *None*, each env
            resets with its existing condition.

        Returns
        -------
        obs : dict
            Stacked observations with batch dimension ``num_envs``.
        info : dict
            Auxiliary information (empty).
        """
        all_obs: List[Dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            tc = text_conditions[i] if text_conditions is not None else None
            obs, _ = env.reset(text_condition=tc)
            all_obs.append(obs)

        # Stack observations
        stacked = {
            "z_k": torch.stack([o["z_k"] for o in all_obs], dim=0),
            "k": torch.stack([o["k"] for o in all_obs], dim=0),
            "text_condition": torch.stack(
                [o["text_condition"] for o in all_obs], dim=0
            ),
            "eps_hat_k": torch.stack(
                [o["eps_hat_k"] for o in all_obs], dim=0
            ),
        }
        return stacked, {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Step all environments with batched actions.

        Auto-resets any environment that terminates.

        Parameters
        ----------
        actions : Tensor, shape ``(num_envs, 129)``
            Batched action tensor.

        Returns
        -------
        obs : dict
            Stacked observations with batch dim ``num_envs``.
        rewards : Tensor, shape ``(num_envs,)``
        terminated : Tensor, shape ``(num_envs,)``
            Boolean tensor.
        truncated : Tensor, shape ``(num_envs,)``
            Always zeros (diffusion episodes are not truncated).
        infos : list of dict
        """
        all_obs: List[Dict[str, Any]] = []
        rewards = torch.zeros(self.num_envs)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        infos: List[Dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            obs, r, term, trunc, info = env.step(actions[i])
            all_obs.append(obs)
            rewards[i] = r
            terminated[i] = term
            truncated[i] = trunc
            infos.append(info)

            # Auto-reset terminated environments
            if term or trunc:
                new_obs, _ = env.reset()
                all_obs[-1] = new_obs
                infos[-1]["terminal_observation"] = obs

        # Stack
        stacked = {
            "z_k": torch.stack([o["z_k"] for o in all_obs], dim=0),
            "k": torch.stack([o["k"] for o in all_obs], dim=0),
            "text_condition": torch.stack(
                [o["text_condition"] for o in all_obs], dim=0
            ),
            "eps_hat_k": torch.stack(
                [o["eps_hat_k"] for o in all_obs], dim=0
            ),
        }

        return stacked, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def close(self) -> None:
        """No-op cleanup (no external resources to release)."""
        pass

    @property
    def text_conditions(self) -> List[torch.Tensor]:
        """Return the current text conditions for all environments."""
        return [env.text_condition for env in self.envs]

    def set_text_conditions(self, conditions: List[torch.Tensor]) -> None:
        """Update text conditions for all environments."""
        for env, cond in zip(self.envs, conditions):
            env.text_condition = cond.to(env.device)
