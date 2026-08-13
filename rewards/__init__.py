"""
signrl_diff.rewards
~~~~~~~~~~~~~~~~~~~

Reward models for the SignRL-Diff pipeline.

Public API
----------
- :class:`HierarchicalReward` -- HAR system with dynamic sub-reward gating.
"""

from .har import HierarchicalReward

__all__ = [
    "HierarchicalReward",
]
