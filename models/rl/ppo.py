"""
signrl_diff.models.rl.ppo
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Proximal Policy Optimisation (PPO) trainer with Generalised Advantage
Estimation (GAE) for the SignRL-Diff diffusion fine-tuning loop.

The trainer orchestrates:

1. **Rollout collection** -- run the policy in (possibly vectorised)
   diffusion environments and store transitions.
2. **GAE computation** -- estimate advantages and lambda-returns from
   stored rewards and value predictions.
3. **PPO clipped-objective update** -- multiple epochs of mini-batch SGD
   on the surrogate objective, value loss, and entropy bonus.

References
----------
* Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
* Schulman et al., "High-Dimensional Continuous Control Using Generalized
  Advantage Estimation", 2016.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .policy import PolicyNetwork
from .value import ValueNetwork


# ======================================================================
# Environment Protocol
# ======================================================================

@runtime_checkable
class DiffusionEnv(Protocol):
    """Minimal protocol that an environment must satisfy to be used with
    :class:`PPOTrainer`.

    The environment wraps one step of the diffusion denoising process.
    Observations are returned as a dictionary with keys:

    * ``"z_k"``   -- ``(T, C, H, W)`` noisy latent
    * ``"k"``     -- ``int`` timestep index
    * ``"text_condition"`` -- ``(L, D_text)`` text embedding
    * ``"eps_hat_k"`` -- ``(T, C, H, W)`` UNet noise prediction
    """

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return ``(observation, info)``."""
        ...

    def step(
        self, action: torch.Tensor
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Return ``(observation, reward, terminated, truncated, info)``."""
        ...


@runtime_checkable
class VecDiffusionEnv(Protocol):
    """Protocol for a vectorised (batched) environment."""

    num_envs: int

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return batched ``(observation, info)``."""
        ...

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Return ``(observation, reward, terminated, truncated, info)``
        where reward / terminated / truncated are 1-D tensors of shape
        ``(num_envs,)``."""
        ...


# ======================================================================
# Rollout Buffer
# ======================================================================

@dataclass
class RolloutBuffer:
    """Stores one rollout of transitions collected by :meth:`PPOTrainer.collect_rollout`.

    All per-transition fields are stored as Python lists and converted to
    batched tensors when :meth:`to_tensors` is called.
    """

    # State components (stored per-transition)
    z_k_list: List[torch.Tensor] = field(default_factory=list)
    k_list: List[torch.Tensor] = field(default_factory=list)
    text_list: List[torch.Tensor] = field(default_factory=list)
    eps_hat_list: List[torch.Tensor] = field(default_factory=list)

    # Action / policy quantities
    actions: List[torch.Tensor] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)

    # Environment outputs
    rewards: List[torch.Tensor] = field(default_factory=list)
    dones: List[torch.Tensor] = field(default_factory=list)

    # Terminal value (for bootstrapping GAE)
    last_value: Optional[torch.Tensor] = None

    @property
    def size(self) -> int:
        """Number of transitions stored."""
        return len(self.actions)

    def clear(self) -> None:
        """Reset all buffers."""
        self.z_k_list.clear()
        self.k_list.clear()
        self.text_list.clear()
        self.eps_hat_list.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.last_value = None

    def to_tensors(
        self, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Stack stored lists into batched tensors on *device*.

        Returns
        -------
        dict with keys:
            ``z_k``, ``k``, ``text_condition``, ``eps_hat_k``,
            ``actions``, ``log_probs``, ``values``, ``rewards``, ``dones``
        """
        return {
            "z_k": torch.stack(self.z_k_list, dim=0).to(device),
            "k": torch.stack(self.k_list, dim=0).to(device),
            "text_condition": torch.stack(self.text_list, dim=0).to(device),
            "eps_hat_k": torch.stack(self.eps_hat_list, dim=0).to(device),
            "actions": torch.stack(self.actions, dim=0).to(device),
            "log_probs": torch.stack(self.log_probs, dim=0).to(device),
            "values": torch.stack(self.values, dim=0).to(device),
            "rewards": torch.stack(self.rewards, dim=0).to(device),
            "dones": torch.stack(self.dones, dim=0).to(device),
        }


# ======================================================================
# PPO Trainer
# ======================================================================

class PPOTrainer:
    """Full PPO trainer with GAE for the SignRL-Diff policy.

    Parameters
    ----------
    policy : PolicyNetwork
        The actor network producing diagonal-Gaussian actions.
    value_fn : ValueNetwork
        The critic network producing scalar state-value estimates.
    lr : float
        Learning rate for both networks (separate optimisers).
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE exponential weighting parameter lambda.
    clip_epsilon : float
        PPO clipping parameter epsilon.
    value_coeff : float
        Weight ``c_1`` on the value loss.
    entropy_coeff : float
        Weight ``c_2`` on the entropy bonus.
    num_epochs : int
        Number of passes over the rollout data per update.
    mini_batch_size : int
        Mini-batch size for SGD.
    max_grad_norm : float
        Gradient clipping threshold (applied to each network).
    device : str or torch.device
        Device for training tensors.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        value_fn: ValueNetwork,
        lr: float = 3e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        num_epochs: int = 4,
        mini_batch_size: int = 64,
        max_grad_norm: float = 0.5,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.value_fn = value_fn.to(device)

        self.gamma: float = gamma
        self.gae_lambda: float = gae_lambda
        self.clip_epsilon: float = clip_epsilon
        self.value_coeff: float = value_coeff
        self.entropy_coeff: float = entropy_coeff
        self.num_epochs: int = num_epochs
        self.mini_batch_size: int = mini_batch_size
        self.max_grad_norm: float = max_grad_norm
        self.device: torch.device = torch.device(device)

        # Separate optimisers for actor and critic
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, eps=1e-5
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_fn.parameters(), lr=lr, eps=1e-5
        )

        # Keep a snapshot of the old policy weights for importance sampling
        # diagnostics.  The actual PPO ratio uses stored log-probs from the
        # rollout buffer (standard practice), but the frozen copy is
        # available for offline evaluation or alternative ratio computation.
        self._old_policy_state_dict: Dict[str, torch.Tensor] = {}

        # Active rollout buffer
        self._buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    # Rollout Collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_rollout(
        self,
        env: Union[DiffusionEnv, VecDiffusionEnv],
        num_steps: int = 50,
    ) -> RolloutBuffer:
        """Collect *num_steps* of experience by running the current policy.

        Supports both single environments (:class:`DiffusionEnv`) and
        vectorised environments (:class:`VecDiffusionEnv`).

        Before collection, the current policy weights are snapshotted
        into ``self._old_policy_state_dict`` for importance-sampling
        reference.

        Parameters
        ----------
        env : DiffusionEnv or VecDiffusionEnv
            The environment to interact with.
        num_steps : int
            Number of environment steps to collect.

        Returns
        -------
        RolloutBuffer
            Populated buffer with stored transitions.
        """
        self.policy.eval()
        self.value_fn.eval()

        # Snapshot old policy for importance-sampling reference
        self._old_policy_state_dict = copy.deepcopy(
            self.policy.state_dict()
        )

        self._buffer.clear()

        is_vec = isinstance(env, VecDiffusionEnv)
        num_envs = getattr(env, "num_envs", 1)

        # ------ Initial reset ------
        obs, info = env.reset()
        z_k = obs["z_k"].to(self.device)
        k = obs["k"].to(self.device) if isinstance(obs["k"], torch.Tensor) \
            else torch.tensor(obs["k"], device=self.device)
        text_cond = obs["text_condition"].to(self.device)
        eps_hat = obs["eps_hat_k"].to(self.device)

        for step_idx in range(num_steps):
            # Build state tuple
            state = (z_k, k, eps_hat)

            # Forward policy and value networks
            action, log_prob = self.policy(state, text_cond)
            value = self.value_fn(state, text_cond)

            # Detach for environment interaction
            action_np = action.cpu()
            log_prob_detached = log_prob.detach()
            value_detached = value.detach()

            # Step environment
            if is_vec:
                obs, reward, terminated, truncated, infos = env.step(action_np)
                done = (terminated | truncated).float()
            else:
                obs, reward_scalar, terminated, truncated, info = env.step(
                    action_np.squeeze(0)
                )
                reward = torch.tensor(
                    [reward_scalar], dtype=torch.float32, device=self.device
                )
                done = torch.tensor(
                    [float(terminated or truncated)],
                    dtype=torch.float32,
                    device=self.device,
                )

            # Store transitions -- handle both single and vectorised
            if is_vec:
                for env_idx in range(num_envs):
                    self._buffer.z_k_list.append(z_k[env_idx].cpu())
                    self._buffer.k_list.append(
                        k[env_idx].cpu() if k.dim() > 0 else k.cpu()
                    )
                    self._buffer.text_list.append(text_cond[env_idx].cpu())
                    self._buffer.eps_hat_list.append(eps_hat[env_idx].cpu())
                    self._buffer.actions.append(action[env_idx].cpu())
                    self._buffer.log_probs.append(log_prob_detached[env_idx].cpu())
                    self._buffer.values.append(value_detached[env_idx].cpu())
                    self._buffer.rewards.append(reward[env_idx].cpu())
                    self._buffer.dones.append(done[env_idx].cpu())
            else:
                self._buffer.z_k_list.append(z_k.squeeze(0).cpu())
                self._buffer.k_list.append(
                    k.squeeze(0).cpu() if k.dim() > 0 else k.cpu()
                )
                self._buffer.text_list.append(text_cond.squeeze(0).cpu())
                self._buffer.eps_hat_list.append(eps_hat.squeeze(0).cpu())
                self._buffer.actions.append(action.squeeze(0).cpu())
                self._buffer.log_probs.append(log_prob_detached.squeeze(0).cpu())
                self._buffer.values.append(value_detached.squeeze(0).cpu())
                self._buffer.rewards.append(reward.squeeze(0).cpu())
                self._buffer.dones.append(done.squeeze(0).cpu())

            # Update current observation for next step
            z_k = obs["z_k"].to(self.device)
            k = obs["k"].to(self.device) if isinstance(obs["k"], torch.Tensor) \
                else torch.tensor(obs["k"], device=self.device)
            text_cond = obs["text_condition"].to(self.device)
            eps_hat = obs["eps_hat_k"].to(self.device)

            # Auto-reset handling for vectorised envs
            if is_vec and done.any():
                # The env is assumed to auto-reset; the new observation is
                # already in `obs` from the step call.
                pass

        # Bootstrap value for the final state (used by GAE)
        final_state = (z_k, k, eps_hat)
        self._buffer.last_value = self.value_fn(
            final_state, text_cond
        ).detach().cpu()

        self.policy.train()
        self.value_fn.train()

        return self._buffer

    # ------------------------------------------------------------------
    # GAE Computation
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalised Advantage Estimates and lambda-returns.

        Given a sequence of T transitions, the TD error at step *t* is:

            delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t

        and the GAE is accumulated backwards:

            A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

        The lambda-return is then:

            G_t = A_t + V_t

        Parameters
        ----------
        rewards : Tensor, shape ``(T,)``
            Per-step rewards.
        values : Tensor, shape ``(T,)``
            Per-step value predictions ``V(s_t)``.
        dones : Tensor, shape ``(T,)``
            Episode termination flags (0 or 1).
        last_value : Tensor, scalar
            Value estimate for the state *after* the final transition
            (bootstrap value).

        Returns
        -------
        advantages : Tensor, shape ``(T,)``
            GAE advantages ``A_hat_t``.
        returns : Tensor, shape ``(T,)``
            Lambda-returns ``G_hat_t = A_hat_t + V_t``.
        """
        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)

        # Initialise the running advantage
        gae = torch.zeros(1, dtype=rewards.dtype, device=rewards.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            delta = (
                rewards[t]
                + self.gamma * next_value * (1.0 - dones[t])
                - values[t]
            )

            # A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------

    def update(
        self,
        rollout_data: Optional[RolloutBuffer] = None,
    ) -> Dict[str, float]:
        """Perform a PPO clipped-objective update using collected rollout data.

        The total loss is:

            L = -L_PPO + c_1 * L_V - c_2 * H[pi]

        where:
            - L_PPO = E[min(rho * A_hat, clip(rho, 1-eps, 1+eps) * A_hat)]
            - L_V   = E[(V(s) - G_hat)^2]
            - H[pi]  = E[entropy of pi]

        Parameters
        ----------
        rollout_data : RolloutBuffer, optional
            If *None*, uses the buffer from the most recent
            :meth:`collect_rollout` call.

        Returns
        -------
        dict
            Training statistics with keys:
            ``"policy_loss"``, ``"value_loss"``, ``"entropy"``,
            ``"total_loss"``, ``"approx_kl"``, ``"clip_fraction"``.
        """
        if rollout_data is None:
            rollout_data = self._buffer

        assert rollout_data.size > 0, "Rollout buffer is empty; collect a rollout first."

        # Convert buffer to batched tensors
        data = rollout_data.to_tensors(self.device)

        z_k = data["z_k"]
        k = data["k"]
        text_cond = data["text_condition"]
        eps_hat = data["eps_hat_k"]
        old_actions = data["actions"]
        old_log_probs = data["log_probs"]
        values = data["values"]
        rewards = data["rewards"]
        dones = data["dones"]

        # Compute GAE advantages and returns
        last_val = rollout_data.last_value.to(self.device) if rollout_data.last_value is not None \
            else torch.zeros(1, device=self.device)

        advantages, returns = self.compute_gae(rewards, values, dones, last_val)

        # Normalise advantages (stabilises training)
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        num_samples = z_k.shape[0]
        state = (z_k, k, eps_hat)

        # Accumulators for logging
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss_acc = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        num_updates = 0

        for _epoch in range(self.num_epochs):
            # Shuffle indices for mini-batch sampling
            indices = torch.randperm(num_samples, device=self.device)

            for start in range(0, num_samples, self.mini_batch_size):
                end = min(start + self.mini_batch_size, num_samples)
                mb_idx = indices[start:end]

                # Extract mini-batch
                mb_state = (
                    z_k[mb_idx],
                    k[mb_idx],
                    eps_hat[mb_idx],
                )
                mb_text = text_cond[mb_idx]
                mb_actions = old_actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # ---- Policy forward ----
                new_log_probs, entropy, _ = self.policy.evaluate(
                    mb_state, mb_text, mb_actions
                )

                # ---- Value forward ----
                new_values = self.value_fn(mb_state, mb_text)

                # ---- Importance sampling ratio ----
                # rho = pi_new(a|s) / pi_old(a|s)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                # ---- Clipped surrogate objective ----
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon,
                    )
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value loss ----
                value_loss = F.mse_loss(new_values, mb_returns)

                # ---- Entropy bonus ----
                entropy_mean = entropy.mean()

                # ---- Total loss ----
                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    - self.entropy_coeff * entropy_mean
                )

                # ---- Back-propagation ----
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.value_fn.parameters(), self.max_grad_norm
                )

                self.policy_optimizer.step()
                self.value_optimizer.step()

                # ---- Logging ----
                with torch.no_grad():
                    approx_kl = (
                        (ratio - 1.0) - log_ratio
                    ).mean().item()
                    clip_frac = (
                        (ratio - 1.0).abs() > self.clip_epsilon
                    ).float().mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_loss_acc += loss.item()
                total_approx_kl += approx_kl
                total_clip_frac += clip_frac
                num_updates += 1

        # Average over all mini-batch updates
        stats = {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "total_loss": total_loss_acc / max(num_updates, 1),
            "approx_kl": total_approx_kl / max(num_updates, 1),
            "clip_fraction": total_clip_frac / max(num_updates, 1),
        }

        # Refresh old-policy snapshot after the update
        self._old_policy_state_dict = copy.deepcopy(
            self.policy.state_dict()
        )

        return stats

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_old_policy(self) -> PolicyNetwork:
        """Return a **new** :class:`PolicyNetwork` initialised with the
        weights from before the most recent rollout collection.

        Useful for offline importance-sampling evaluation or diagnostics.
        """
        old_policy = copy.deepcopy(self.policy)
        old_policy.load_state_dict(self._old_policy_state_dict)
        old_policy.eval()
        return old_policy

    def save_checkpoint(self, path: str) -> None:
        """Save trainer state (both networks + optimisers) to *path*."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_fn_state_dict": self.value_fn.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Restore trainer state from a checkpoint at *path*."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.value_fn.load_state_dict(ckpt["value_fn_state_dict"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        self.value_optimizer.load_state_dict(ckpt["value_optimizer"])
