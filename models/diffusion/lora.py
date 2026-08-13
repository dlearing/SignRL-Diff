"""
signrl_diff.models.diffusion.lora
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Low-Rank Adaptation (LoRA) wrapper for ``nn.Linear`` layers.

Reference
---------
Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.

In the SignRL-Diff pipeline every attention projection (Q, K, V, O) inside
the VideoUNet is replaced with a ``LoRALinear`` so that the frozen base
weights remain untouched while a small number of trainable low-rank
parameters steer the model towards the sign-language domain.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank residual.

    Forward pass::

        y = W_frozen(x) + (alpha / r) * B(A(x))

    where ``A ∈ R^{in_features × r}`` (kaiming-uniform init) and
    ``B ∈ R^{r × out_features}`` (zero init), so that the LoRA branch
    starts as an identity-zero contribution.

    Parameters
    ----------
    original : nn.Linear
        The pre-trained linear layer.  Its parameters are **frozen**
        (``requires_grad=False``) and it is never updated by the optimiser.
    rank : int, default 16
        Rank of the low-rank decomposition.
    alpha : float, default 16.0
        Scaling constant applied to the LoRA branch.  The effective scale
        is ``alpha / rank``.

    Attributes
    ----------
    lora_A : nn.Parameter
        Down-projection matrix, shape ``(in_features, rank)``.
    lora_B : nn.Parameter
        Up-projection matrix, shape ``(rank, out_features)``.
    merged : bool
        ``True`` after :py:meth:`merge` has been called and before
        :py:meth:`reset`.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()

        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")

        self.rank: int = rank
        self.alpha: float = alpha
        self.scaling: float = alpha / rank
        self.merged: bool = False

        # ------------------------------------------------------------------
        # Freeze the original linear layer but keep it as a sub-module so
        # its weight/bias are part of ``state_dict``.
        # ------------------------------------------------------------------
        self.linear: nn.Linear = copy.deepcopy(original)
        for param in self.linear.parameters():
            param.requires_grad = False

        in_features: int = self.linear.in_features
        out_features: int = self.linear.out_features

        # ------------------------------------------------------------------
        # Low-rank adapter matrices
        # ------------------------------------------------------------------
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # A: kaiming-uniform (same as PyTorch default for Linear)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))
        # B: zeros — the LoRA branch contributes nothing at initialisation
        nn.init.zeros_(self.lora_B)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA-augmented linear projection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose last dimension equals ``in_features``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(*, out_features)``.
        """
        # Base (frozen) projection
        base_out: torch.Tensor = self.linear(x)

        if self.merged:
            # After merge(), the LoRA delta is already baked into
            # ``self.linear.weight`` — no extra computation needed.
            return base_out

        # Low-rank residual: x @ A @ B  scaled by alpha/r
        lora_out: torch.Tensor = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

    # ------------------------------------------------------------------
    # Merge / Reset helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the LoRA delta into the frozen base weight.

        After merging, the forward path is equivalent to a plain
        ``nn.Linear`` (no extra FLOPs from the adapter branch).
        Call :py:meth:`reset` to undo this operation.

        Raises
        ------
        RuntimeError
            If the adapters have already been merged.
        """
        if self.merged:
            raise RuntimeError("LoRA weights are already merged.")

        # delta_W = (alpha / r) * B^T @ A^T  →  shape (out, in)
        # Note: lora_A is (in, r), lora_B is (r, out).
        # delta_W should be (out, in) to match self.linear.weight shape.
        delta_w: torch.Tensor = (
            self.lora_B.t() @ self.lora_A.t()
        ) * self.scaling

        self.linear.weight.data += delta_w
        self.merged = True

    @torch.no_grad()
    def reset(self) -> None:
        """Zero-out the adapter matrices and un-merge if necessary.

        After calling ``reset()`` the LoRA branch again contributes
        nothing, and the module is in the same state as immediately
        after ``__init__`` (aside from any merged-then-unmerged weight
        changes, which are undone here).
        """
        if self.merged:
            # Subtract the delta that was previously added.
            delta_w: torch.Tensor = (
                self.lora_B.t() @ self.lora_A.t()
            ) * self.scaling
            self.linear.weight.data -= delta_w
            self.merged = False

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_B)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Provide a readable summary in ``print(model)`` output."""
        return (
            f"in_features={self.linear.in_features}, "
            f"out_features={self.linear.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"merged={self.merged}"
        )
