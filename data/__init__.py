"""
signrl_diff.data
~~~~~~~~~~~~~~~~

Dataset classes for sign language video data.

Public API
----------
- :class:`SignLanguageVideoDataset` -- Video + text dataset.
- :func:`build_preference_pairs`    -- Generate pseudo-preference pairs.
"""

from .dataset import SignLanguageVideoDataset, build_preference_pairs

__all__ = [
    "SignLanguageVideoDataset",
    "build_preference_pairs",
]
