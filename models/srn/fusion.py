"""
Sub-Reward Network — Fusion Module (SignRL-Diff).

Combines the four sub-reward streams (GCN, TCN, Semantic, Hand) into a
single scalar reward in [-1, 1] via a fusion MLP.  Also provides the
listwise ranking loss (Bradley-Terry) used during training.

Inputs:
    video       : R^{B, T, 3, 256, 256}  (generated video frames)
    keypoints   : R^{B, T, 148, 3}       (3-D SMPL-X joints from FrankMocap)
    text_emb    : R^{B, 512}             (CLIP text embedding)
    video_emb   : R^{B, 1024}            (VideoSwin pooled features)
    hand_crops  : R^{B, T, 2, 3, 128, 128}  (left + right hand crops)

Output:
    reward      : R^{B}  scalar in [-1, 1]
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_stream import PoseGCNStream
from .tcn_stream import TemporalCoherenceStream
from .semantic_stream import SemanticAlignmentStream
from .hand_stream import HandArticulationStream


# ---------------------------------------------------------------------------
# Feature extraction helper (video backbone)
# ---------------------------------------------------------------------------

class _VideoFeatureExtractor(nn.Module):
    """Lightweight 3-D CNN that extracts a 1024-d video embedding.

    This module replaces the external VideoSwin backbone during
    end-to-end training.  It applies a series of 3-D convolutions
    with global average pooling to produce a fixed-size embedding.

    Args:
        in_channels: Number of input channels (3 for RGB).
        feature_dim: Output embedding dimensionality.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 1024) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # Block 2
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # Block 3
            nn.Conv3d(128, 256, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # Block 4
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            # Block 5
            nn.Conv3d(512, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: ``(B, T, 3, H, W)`` — needs permutation to ``(B, 3, T, H, W)``
                   for Conv3d.

        Returns:
            Video embedding of shape ``(B, feature_dim)``.
        """
        # (B, T, C, H, W) → (B, C, T, H, W)
        x = video.permute(0, 2, 1, 3, 4).contiguous()
        x = self.features(x)
        x = self.pool(x)                # (B, C, 1, 1, 1)
        x = x.flatten(start_dim=1)      # (B, C)
        return x


# ---------------------------------------------------------------------------
# Hand Crop Extractor
# ---------------------------------------------------------------------------

class _HandCropExtractor(nn.Module):
    """Extracts hand crops from the full video using keypoint-guided regions.

    When explicit hand crops are not provided, this module crops around the
    wrist/hand keypoints (joints 11 and 14 in SMPL-X body skeleton) with a
    fixed-size bounding box.

    For simplicity, if explicit crops are passed the module is bypassed; this
    class exists to provide a fallback path.

    Args:
        crop_size: Spatial size of each hand crop (square).
    """

    def __init__(self, crop_size: int = 128) -> None:
        super().__init__()
        self.crop_size = crop_size

    def forward(
        self,
        video: torch.Tensor,
        keypoints: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hand crops from the video frames.

        Uses keypoints 11 (left wrist) and 14 (right wrist) projected to
        image coordinates to define crop centres.  Falls back to centre
        crops when keypoints are out-of-range.

        Args:
            video: ``(B, T, 3, H, W)`` video tensor.
            keypoints: ``(B, T, 148, 3)`` keypoint tensor.

        Returns:
            Hand crops of shape ``(B, T, 2, 3, crop_size, crop_size)``.
        """
        B, T, C, H, W = video.shape
        half = self.crop_size // 2

        # Use wrist joints (11 = left, 14 = right) as crop centres.
        # Keypoints are in normalised [-1, 1] or pixel space — map to [0, W) and [0, H).
        left_wrist = keypoints[:, :, 11, :2]   # (B, T, 2)  — (x, y)
        right_wrist = keypoints[:, :, 14, :2]  # (B, T, 2)

        # Assume keypoints are in pixel-like coordinates relative to 256x256.
        # Clamp to valid range.
        left_cx = left_wrist[..., 0].long().clamp(min=half, max=W - half)
        left_cy = left_wrist[..., 1].long().clamp(min=half, max=H - half)
        right_cx = right_wrist[..., 0].long().clamp(min=half, max=W - half)
        right_cy = right_wrist[..., 1].long().clamp(min=half, max=H - half)

        hands = []
        for hand_cx, hand_cy in [(left_cx, left_cy), (right_cx, right_cy)]:
            crops = []
            for b in range(B):
                frame_crops = []
                for t in range(T):
                    cx = hand_cx[b, t].item()
                    cy = hand_cy[b, t].item()
                    crop = video[b, t, :, cy - half: cy + half, cx - half: cx + half]
                    frame_crops.append(crop)
                crops.append(torch.stack(frame_crops))  # (T, 3, cs, cs)
            hands.append(torch.stack(crops))  # (B, T, 3, cs, cs)

        # Stack left and right: (B, T, 2, 3, cs, cs)
        hand_crops = torch.stack(hands, dim=2)
        return hand_crops


# ---------------------------------------------------------------------------
# Listwise Ranking Loss (Bradley-Terry)
# ---------------------------------------------------------------------------

def listwise_ranking_loss(
    rewards: torch.Tensor,
    rankings: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Plackett-Luce / Bradley-Terry listwise ranking loss.

    L = -E[ log( exp(r_i / tau) / sum_j exp(r_j / tau) ) ]

    Given a batch of reward scores and their ground-truth rankings, this
    loss encourages the model to assign higher rewards to better-ranked
    samples.

    Args:
        rewards: Predicted reward scores of shape ``(B,)`` or ``(B, G)``
            where *G* is the group size for listwise comparison.
        rankings: Ground-truth ranking indices of shape ``(B, G)`` — each
            row is a permutation of ``[0, G)`` where index 0 is the best.
            If *rankings* is 1-D ``(B,)`` it is treated as quality scores
            (higher = better) and the loss is computed pairwise across the
            batch.
        temperature: Temperature scaling factor *tau* (lower = sharper).

    Returns:
        Scalar loss tensor.
    """
    if rewards.dim() == 1:
        # Pairwise interpretation: for each pair (i, j) where rank_i < rank_j,
        # we want reward_i > reward_j.  Use the soft-margin formulation.
        # L = -mean( log( sigmoid( (r_i - r_j) / tau ) ) )  for i ranked higher
        B = rewards.size(0)
        if B < 2:
            return torch.tensor(0.0, device=rewards.device, requires_grad=True)

        # Normalise rankings to [0, 1]
        rank_norm = rankings.float()
        rank_norm = (rank_norm - rank_norm.min()) / (rank_norm.max() - rank_norm.min() + 1e-8)

        # Pairwise differences
        diff = rewards.unsqueeze(1) - rewards.unsqueeze(0)  # (B, B)
        rank_diff = rank_norm.unsqueeze(1) - rank_norm.unsqueeze(0)  # (B, B)

        # Only consider pairs where rank_i > rank_j (i is worse)
        mask = (rank_diff > 0).float()
        logits = diff / temperature

        # Soft-margin: -log(sigmoid(r_better - r_worse))
        loss = -F.logsigmoid(logits) * mask
        # Normalise by number of valid pairs
        num_pairs = mask.sum().clamp(min=1.0)
        return loss.sum() / num_pairs

    else:
        # Group listwise: rewards is (B, G), rankings is (B, G)
        # For each group, compute the Plackett-Luce log-likelihood
        scaled = rewards / temperature  # (B, G)
        log_probs = F.log_softmax(scaled, dim=-1)  # (B, G)

        # The loss is the negative log-likelihood of the correct ranking order.
        # We gather the log-prob of the top-ranked item, then remove it and
        # repeat (sequential log-likelihood of the permutation).
        B, G = rewards.shape
        total_loss = torch.tensor(0.0, device=rewards.device)

        remaining = torch.arange(G, device=rewards.device).unsqueeze(0).expand(B, -1)
        for step in range(G - 1):
            # Index of the item ranked 'step'-th in each group
            best_idx = rankings[:, step]  # (B,)
            # Gather log-prob of the best item among remaining
            step_log_probs = F.log_softmax(
                torch.gather(scaled, 1, remaining), dim=-1
            )  # (B, remaining_size)
            # Map best_idx to its position in 'remaining'
            pos = (remaining == best_idx.unsqueeze(1)).long().argmax(dim=1)  # (B,)
            total_loss = total_loss - step_log_probs.gather(1, pos.unsqueeze(1)).squeeze(1).mean()
            # Remove the selected item from remaining
            mask = remaining != best_idx.unsqueeze(1)
            remaining = remaining[mask].reshape(B, -1)

        return total_loss


# ---------------------------------------------------------------------------
# Sub-Reward Network (Fusion)
# ---------------------------------------------------------------------------

class SubRewardNetwork(nn.Module):
    """Sub-Reward Network (SRN) — fuses four evaluation streams.

    The SRN runs four independent streams on the generated sign-language
    video and fuses their outputs into a single scalar reward in [-1, 1].

    Streams:
        1. **Pose GCN** — body pose quality from 3-D keypoints → f_pose (128-d)
        2. **Temporal TCN** — motion smoothness from pose deltas → f_temp (256-d)
        3. **Semantic** — text-video alignment via CLIP similarity → f_sem (1-d)
        4. **Hand CNN** — hand articulation from cropped regions → f_hand (128-d)

    Fusion:
        concat([f_pose, f_temp, f_sem, f_hand]) = R^{513}
        → MLP(513 → 256 → 128 → 1) with ReLU, Dropout(0.3), final Tanh

    Args:
        num_joints: Number of skeleton joints for the GCN stream.
        video_emb_dim: Dimensionality of video embeddings (for VideoSwin).
        use_video_extractor: If *True*, a built-in 3-D CNN extracts video
            embeddings from raw frames (no external VideoSwin needed).
        use_hand_extractor: If *True*, hand crops are extracted from the
            video using keypoint-guided regions when not provided.
    """

    def __init__(
        self,
        num_joints: int = 148,
        video_emb_dim: int = 1024,
        use_video_extractor: bool = True,
        use_hand_extractor: bool = True,
    ) -> None:
        super().__init__()

        # --- Individual streams ---
        self.gcn_stream = PoseGCNStream(num_joints=num_joints)
        self.tcn_stream = TemporalCoherenceStream()
        self.semantic_stream = SemanticAlignmentStream(video_dim=video_emb_dim)
        self.hand_stream = HandArticulationStream()

        # --- Optional feature extractors ---
        self.use_video_extractor = use_video_extractor
        if use_video_extractor:
            self.video_extractor = _VideoFeatureExtractor(feature_dim=video_emb_dim)

        self.use_hand_extractor = use_hand_extractor
        if use_hand_extractor:
            self.hand_extractor = _HandCropExtractor(crop_size=128)

        # --- Fusion MLP ---
        # Concatenated feature size: 128 + 256 + 1 + 128 = 513
        fused_dim = 513
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
        )
        self.tanh = nn.Tanh()

    def forward(
        self,
        video: torch.Tensor,
        keypoints: torch.Tensor,
        text_emb: torch.Tensor,
        video_emb: torch.Tensor | None = None,
        hand_crops: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the fused sub-reward for a batch of generated videos.

        Args:
            video: Generated video tensor ``(B, T, 3, 256, 256)``.
            keypoints: 3-D keypoints ``(B, T, 148, 3)`` from FrankMocap.
            text_emb: CLIP text embeddings ``(B, 512)``.
            video_emb: Optional pre-computed video features ``(B, 1024)``.
                If *None* and ``use_video_extractor`` is *True*, features
                are extracted from ``video``.
            hand_crops: Optional pre-extracted hand crops
                ``(B, T, 2, 3, 128, 128)``.  If *None* and
                ``use_hand_extractor`` is *True*, crops are extracted
                from ``video`` using ``keypoints``.

        Returns:
            Reward tensor of shape ``(B,)`` with values in [-1, 1].
        """
        sub = self.compute_sub_rewards(
            video, keypoints, text_emb, video_emb, hand_crops,
        )

        # Concatenate all stream outputs: (B, 513)
        fused = torch.cat([
            sub["f_pose"],    # (B, 128)
            sub["f_temp"],    # (B, 256)
            sub["f_sem"].unsqueeze(1),  # (B, 1)
            sub["f_hand"],    # (B, 128)
        ], dim=-1)

        # Fusion MLP + Tanh
        reward = self.fusion_mlp(fused).squeeze(-1)  # (B,)
        reward = self.tanh(reward)                     # [-1, 1]
        return reward

    def compute_sub_rewards(
        self,
        video: torch.Tensor,
        keypoints: torch.Tensor,
        text_emb: torch.Tensor,
        video_emb: torch.Tensor | None = None,
        hand_crops: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Run all four streams and return individual sub-rewards.

        This is useful for monitoring, auxiliary losses, and debugging.

        Args:
            video: ``(B, T, 3, 256, 256)``
            keypoints: ``(B, T, 148, 3)``
            text_emb: ``(B, 512)``
            video_emb: Optional ``(B, 1024)``
            hand_crops: Optional ``(B, T, 2, 3, 128, 128)``

        Returns:
            Dictionary with keys ``f_pose``, ``f_temp``, ``f_sem``,
            ``f_hand`` and their respective tensor outputs.
        """
        # --- 1. Pose GCN stream ---
        f_pose = self.gcn_stream(keypoints)  # (B, 128)

        # --- 2. Temporal coherence stream ---
        # Compute frame-to-frame keypoint differences
        pose_diff = keypoints[:, 1:, :, :] - keypoints[:, :-1, :, :]  # (B, T-1, 148, 3)
        f_temp = self.tcn_stream(pose_diff)  # (B, 256)

        # --- 3. Semantic alignment stream ---
        if video_emb is None:
            if self.use_video_extractor:
                video_emb = self.video_extractor(video)  # (B, 1024)
            else:
                raise ValueError(
                    "video_emb is None and use_video_extractor is False. "
                    "Provide video embeddings or enable the built-in extractor."
                )
        f_sem = self.semantic_stream(text_emb, video_emb)  # (B,)

        # --- 4. Hand articulation stream ---
        if hand_crops is None:
            if self.use_hand_extractor:
                hand_crops = self.hand_extractor(video, keypoints)
            else:
                raise ValueError(
                    "hand_crops is None and use_hand_extractor is False. "
                    "Provide hand crops or enable the built-in extractor."
                )
        f_hand = self.hand_stream(hand_crops)  # (B, 128)

        return {
            "f_pose": f_pose,
            "f_temp": f_temp,
            "f_sem": f_sem,
            "f_hand": f_hand,
        }

    @staticmethod
    def ranking_loss(
        rewards: torch.Tensor,
        rankings: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Convenience wrapper around :func:`listwise_ranking_loss`.

        Args:
            rewards: Predicted rewards ``(B,)`` or ``(B, G)``.
            rankings: Ground-truth rankings ``(B,)`` or ``(B, G)``.
            temperature: Temperature scaling.

        Returns:
            Scalar loss.
        """
        return listwise_ranking_loss(rewards, rankings, temperature)
