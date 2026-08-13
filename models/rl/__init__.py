"""
signrl_diff.models.rl
~~~~~~~~~~~~~~~~~~~~~~

Reinforcement-learning components for the SignRL-Diff pipeline.

Public API
----------
- :class:`PolicyNetwork` -- Lightweight transformer policy with cross-attention.
- :class:`ValueNetwork`  -- Critic network producing scalar state-value estimates.
- :class:`PPOTrainer`    -- PPO trainer with GAE for diffusion fine-tuning.
"""

from .policy import PolicyNetwork
from .value import ValueNetwork
from .ppo import PPOTrainer

__all__ = [
    "PolicyNetwork",
    "ValueNetwork",
    "PPOTrainer",
]
