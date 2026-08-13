"""
signrl_diff.utils.helpers
~~~~~~~~~~~~~~~~~~~~~~~~~

Utility functions for the SignRL-Diff pipeline: seeding, checkpointing,
parameter counting, and action-to-latent-space projection.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# Seeding
# ======================================================================

def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility across all libraries.

    Seeds Python's ``random``, NumPy, and PyTorch (CPU + CUDA).
    Also enables cuDNN deterministic mode and disables benchmarking
    for fully reproducible convolution results.

    Parameters
    ----------
    seed : int
        The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ======================================================================
# Parameter Counting
# ======================================================================

def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """Count the number of parameters in a model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.
    only_trainable : bool
        If *True*, count only parameters with ``requires_grad=True``.

    Returns
    -------
    int
        Total parameter count.
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ======================================================================
# Checkpoint Save / Load
# ======================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    step: int,
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint to disk.

    The checkpoint contains the model state dict, optimizer state dict
    (if provided), epoch/step counters, and any extra metadata.

    Parameters
    ----------
    model : nn.Module
        The model whose weights should be saved.
    optimizer : Optimizer or None
        The optimizer whose state should be saved (may be *None*).
    epoch : int
        Current epoch number.
    step : int
        Current global training step.
    path : str
        File path for the checkpoint (typically ``*.pt``).
    extra : dict, optional
        Additional key-value pairs to include in the checkpoint.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint from disk.

    Restores model weights and (optionally) optimizer state, then
    returns the remaining metadata (epoch, step, extra keys).

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    model : nn.Module
        Model to load weights into.
    optimizer : Optimizer or None
        Optimizer to load state into (may be *None*).
    device : str or torch.device
        Device to map loaded tensors onto.

    Returns
    -------
    dict
        Metadata dictionary with at least ``"epoch"`` and ``"step"`` keys.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    meta: Dict[str, Any] = {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
    }
    # Forward any extra keys
    for key in checkpoint:
        if key not in ("model_state_dict", "optimizer_state_dict", "epoch", "step"):
            meta[key] = checkpoint[key]

    return meta


# ======================================================================
# Action-to-Latent-Space Projection
# ======================================================================

def action_to_correction(
    action: torch.Tensor,
    w_global: nn.Linear,
    m_hand_left: torch.Tensor,
    m_hand_right: torch.Tensor,
    num_frames: int = 16,
    latent_channels: int = 4,
    latent_hw: int = 32,
) -> torch.Tensor:
    """Project a 129-d action tensor into a latent-space noise correction.

    The action is decomposed as::

        a_k = [global(64) | hand_left(32) | hand_right(32) | scale(1)]

    The global component is projected through a learned linear layer
    ``W_global`` to produce a full latent-shaped correction tensor.
    The hand components are projected through sparse hand-region masks
    ``M_hand_left`` and ``M_hand_right`` and added to the corresponding
    spatial regions of the correction.

    Parameters
    ----------
    action : Tensor, shape ``(B, 129)``
        Batched action tensor.
    w_global : nn.Linear
        Learned projection from 64-d global action to latent flat dim.
    m_hand_left : Tensor, shape ``(latent_flat, 32)``
        Sparse projection matrix for left-hand correction.
    m_hand_right : Tensor, shape ``(latent_flat, 32)``
        Sparse projection matrix for right-hand correction.
    num_frames : int
        Temporal length *T* of the latent.
    latent_channels : int
        Channel count *C* of the latent.
    latent_hw : int
        Spatial height/width *H* = *W* of the latent.

    Returns
    -------
    eps_corr : Tensor, shape ``(B, T, C, H, W)``
        Latent-space noise correction tensor.
    """
    B = action.shape[0]
    latent_flat = num_frames * latent_channels * latent_hw * latent_hw

    # Decompose action
    a_global = action[:, :64]                     # (B, 64)
    a_hand_left = action[:, 64:96]                # (B, 32)
    a_hand_right = action[:, 96:128]              # (B, 32)
    # a_scale = action[:, 128:129]                # (B, 1) -- handled by caller

    # Global projection: (B, 64) -> (B, latent_flat)
    eps_global = w_global(a_global)               # (B, latent_flat)

    # Hand projections: sparse matrix multiply
    # m_hand_left: (latent_flat, 32), a_hand_left: (B, 32) -> (B, latent_flat)
    eps_hand_l = a_hand_left @ m_hand_left.t()    # (B, latent_flat)
    eps_hand_r = a_hand_right @ m_hand_right.t()  # (B, latent_flat)

    # Combine corrections
    eps_corr_flat = eps_global + eps_hand_l + eps_hand_r  # (B, latent_flat)

    # Reshape to latent shape: (B, T, C, H, W)
    eps_corr = eps_corr_flat.reshape(
        B, num_frames, latent_channels, latent_hw, latent_hw
    )
    return eps_corr


def build_hand_sparse_projection(
    num_frames: int = 16,
    latent_channels: int = 4,
    latent_hw: int = 32,
    hand_action_dim: int = 32,
    hand_region_ratio: float = 0.25,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a sparse projection matrix that maps a hand action vector
    to the hand-region subset of the latent space.

    The matrix has shape ``(latent_flat, hand_action_dim)`` where
    ``latent_flat = T * C * H * W``.  Non-zero entries are concentrated
    in the spatial region corresponding to the hand area (approximately
    the bottom-left or bottom-right quarter of the latent spatial grid).

    Parameters
    ----------
    num_frames : int
        Temporal frames *T*.
    latent_channels : int
        Latent channels *C*.
    latent_hw : int
        Latent spatial size *H* = *W*.
    hand_action_dim : int
        Dimensionality of the hand action component (32).
    hand_region_ratio : float
        Fraction of spatial area allocated to the hand region.
    device : str or torch.device
        Target device for the tensor.

    Returns
    -------
    Tensor, shape ``(latent_flat, hand_action_dim)``
        Sparse projection matrix (not a parameter; used as a fixed mask).
    """
    latent_flat = num_frames * latent_channels * latent_hw * latent_hw
    M = torch.zeros(latent_flat, hand_action_dim, device=device)

    # Define hand region in spatial grid (bottom-right quarter)
    hand_hw = int(latent_hw * hand_region_ratio ** 0.5)
    hand_hw = max(hand_hw, 1)
    h_start = latent_hw - hand_hw
    w_start = latent_hw - hand_hw

    # For each frame and channel, map hand_action_dim elements to the
    # hand spatial region via a round-robin assignment
    region_size = num_frames * latent_channels * hand_hw * hand_hw
    for idx in range(min(hand_action_dim, region_size)):
        # Compute (t, c, h, w) in the hand region
        flat_in_region = idx % region_size
        t = flat_in_region // (latent_channels * hand_hw * hand_hw)
        rem = flat_in_region % (latent_channels * hand_hw * hand_hw)
        c = rem // (hand_hw * hand_hw)
        rem2 = rem % (hand_hw * hand_hw)
        h_local = rem2 // hand_hw
        w_local = rem2 % hand_hw

        h_abs = h_start + h_local
        w_abs = w_start + w_local

        # Flat index in the full latent
        full_flat = (
            t * (latent_channels * latent_hw * latent_hw)
            + c * (latent_hw * latent_hw)
            + h_abs * latent_hw
            + w_abs
        )
        if full_flat < latent_flat:
            # Small random init for diversity
            M[full_flat, idx] = 1.0 / max(hand_action_dim, 1)

    return M
