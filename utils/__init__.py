"""
signrl_diff.utils
~~~~~~~~~~~~~~~~~

Utility functions for the SignRL-Diff pipeline.

Public API
----------
- :func:`set_seed`              -- Reproducible seeding.
- :func:`count_parameters`      -- Model parameter counting.
- :func:`save_checkpoint`       -- Save training checkpoint.
- :func:`load_checkpoint`       -- Load training checkpoint.
- :func:`action_to_correction`  -- Project action to latent-space correction.
- :func:`build_hand_sparse_projection` -- Build sparse hand projection matrix.
"""

from .helpers import (
    set_seed,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    action_to_correction,
    build_hand_sparse_projection,
)

__all__ = [
    "set_seed",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "action_to_correction",
    "build_hand_sparse_projection",
]
