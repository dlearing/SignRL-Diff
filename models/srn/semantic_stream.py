"""
Semantic Alignment Stream for the Sub-Reward Network (SignRL-Diff).

Evaluates cross-modal semantic alignment between the input text prompt
and the generated video using projected cosine similarity in CLIP
embedding space.

Input:  c_text  in R^{B, 512}   (CLIP text embedding)
        c_video in R^{B, 1024}  (VideoSwin / pooled video features)
Output: f_sem   in R^{B}         (cosine similarity scalar in [-1, 1])
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAlignmentStream(nn.Module):
    """Cross-modal CLIP-similarity stream for semantic alignment scoring.

    The stream learns a linear projection ``Proj in R^{512 x 1024}`` that
    maps the video embedding into the CLIP text embedding space.  The
    semantic alignment score is then the cosine similarity between the
    projected video features and the text embedding.

    Architecture:
        1. Linear projection: c_video @ Proj^T  →  R^{B, 512}
        2. Layer normalisation (for stable cosine similarity)
        3. Cosine similarity with c_text

    Args:
        text_dim: Dimensionality of the text embedding (default 512,
            matching CLIP ViT-B/32 text encoder output).
        video_dim: Dimensionality of the video embedding (default 1024,
            matching VideoSwin-Tiny pooled output).
        use_layer_norm: Whether to apply LayerNorm before computing
            cosine similarity (improves training stability).
    """

    def __init__(
        self,
        text_dim: int = 512,
        video_dim: int = 1024,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.video_dim = video_dim

        # Projection matrix: R^{text_dim x video_dim}
        self.proj = nn.Linear(video_dim, text_dim, bias=False)

        # Optional layer norms for both modalities
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.text_ln = nn.LayerNorm(text_dim)
            self.video_ln = nn.LayerNorm(text_dim)  # after projection → text_dim

    def forward(
        self,
        text_emb: torch.Tensor,
        video_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the semantic alignment score.

        Args:
            text_emb: CLIP text embeddings of shape ``(B, 512)``.
            video_emb: Pooled video features of shape ``(B, 1024)``.

        Returns:
            Per-sample cosine similarity ``f_sem`` of shape ``(B,)``,
            with values in ``[-1, 1]``.
        """
        # Project video features into text embedding space
        video_proj = self.proj(video_emb)  # (B, 512)

        # Optional layer normalisation
        if self.use_layer_norm:
            text_emb = self.text_ln(text_emb)
            video_proj = self.video_ln(video_proj)

        # L2-normalise both vectors for cosine similarity
        text_norm = F.normalize(text_emb, p=2, dim=-1)
        video_norm = F.normalize(video_proj, p=2, dim=-1)

        # Cosine similarity → scalar per sample
        f_sem = (text_norm * video_norm).sum(dim=-1)  # (B,)
        return f_sem
