"""
Sub-Reward Network (SRN) package for SignRL-Diff.

This package implements the four-stream Sub-Reward Network that evaluates
generated sign-language videos along complementary axes:

1. **PoseGCNStream** — body pose quality via Graph Convolutional Networks
   on 3-D SMPL-X keypoints (148 joints).
2. **TemporalCoherenceStream** — motion smoothness via dilated causal 1-D
   convolutions on frame-to-frame pose differences.
3. **SemanticAlignmentStream** — cross-modal alignment via projected
   cosine similarity between CLIP text embeddings and video features.
4. **HandArticulationStream** — fine-grained hand evaluation via an
   EfficientNet-B4-style CNN on cropped hand regions.

The **SubRewardNetwork** fusion module combines all four streams into a
single scalar reward in [-1, 1].

Typical usage::

    from signrl_diff.models.srn import SubRewardNetwork

    srn = SubRewardNetwork()
    reward = srn(video, keypoints, text_emb)
"""

from .gcn_stream import PoseGCNStream, build_adjacency_matrix
from .tcn_stream import TemporalCoherenceStream
from .semantic_stream import SemanticAlignmentStream
from .hand_stream import HandArticulationStream
from .fusion import SubRewardNetwork, listwise_ranking_loss

__all__ = [
    "PoseGCNStream",
    "TemporalCoherenceStream",
    "SemanticAlignmentStream",
    "HandArticulationStream",
    "SubRewardNetwork",
    "build_adjacency_matrix",
    "listwise_ranking_loss",
]
