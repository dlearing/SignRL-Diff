"""
GCN Stream for the Sub-Reward Network (SignRL-Diff).

Evaluates body pose quality from 3D SMPL-X keypoints using a Graph
Convolutional Network with symmetric-normalised adjacency.

Input:  key points J_hat in R^{B, T, 148, 3}
Output: f_pose in R^{B, 128}
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SMPL-X skeleton edge list (148 joints)
# ---------------------------------------------------------------------------
# We define a *representative* SMPL-X body skeleton with 148 joints that
# covers: spine/head chain, left & right arms with full hand articulation,
# and left & right legs.  The exact joint ordering follows the common
# FrankMocap / SMPL-X convention (body + left hand + right hand).
# ---------------------------------------------------------------------------

# Body joints: 0-21 (22 joints)
_BODY_EDGES: List[Tuple[int, int]] = [
    # spine chain
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    # head
    (5, 6), (6, 7), (7, 8),
    # left arm (from left shoulder)
    (3, 9), (9, 10), (10, 11),
    # right arm (from right shoulder)
    (3, 12), (12, 13), (13, 14),
    # left leg
    (0, 15), (15, 16), (16, 17),
    # right leg
    (0, 18), (18, 19), (19, 20),
    # extra body (clavicles, eyes, jaw, etc.)
    (5, 21),
]

# Left hand joints: 22-42 (21 joints, indices 22..42)
_LEFT_HAND_EDGES: List[Tuple[int, int]] = [
    # wrist to MCPs
    (11, 22),  # left wrist -> left hand wrist
    (22, 23), (23, 24), (24, 25),          # thumb
    (22, 26), (26, 27), (27, 28), (28, 29),  # index
    (22, 30), (30, 31), (31, 32), (32, 33),  # middle
    (22, 34), (34, 35), (35, 36), (36, 37),  # ring
    (22, 38), (38, 39), (39, 40), (40, 41),  # pinky
    (22, 42),                                   # extra
]

# Right hand joints: 43-63 (21 joints, indices 43..63)
_RIGHT_HAND_EDGES: List[Tuple[int, int]] = [
    (14, 43),  # right wrist -> right hand wrist
    (43, 44), (44, 45), (45, 46),          # thumb
    (43, 47), (47, 48), (48, 49), (49, 50),  # index
    (43, 51), (51, 52), (52, 53), (53, 54),  # middle
    (43, 55), (55, 56), (56, 57), (57, 58),  # ring
    (43, 59), (59, 60), (60, 61), (61, 62),  # pinky
    (43, 63),                                   # extra
]

# Additional face / expression joints: 64-147 (84 joints)
# These correspond to facial landmarks in SMPL-X (jaw, eyes, eyebrows, lips).
# We connect them in a chain starting from the head joint for simplicity,
# and add local lateral connections for nearby landmarks.
_FACE_CHAIN_START = 8   # head tip
_FACE_JOINTS_START = 64
_FACE_JOINTS_END = 147  # inclusive → 84 joints


def _build_face_edges() -> List[Tuple[int, int]]:
    """Build edges for the 84 face landmarks.

    The face landmarks are connected as a chain starting from the head,
    plus short-range lateral connections that approximate facial topology.
    """
    edges: List[Tuple[int, int]] = []
    # Chain from head to first face joint
    edges.append((_FACE_CHAIN_START, _FACE_JOINTS_START))
    # Sequential chain through face landmarks
    for j in range(_FACE_JOINTS_START, _FACE_JOINTS_END):
        edges.append((j, j + 1))
    # Lateral connections (skip-2) for richer face graph
    for j in range(_FACE_JOINTS_START, _FACE_JOINTS_END - 2):
        edges.append((j, j + 2))
    return edges


def build_adjacency_matrix(num_joints: int = 148) -> torch.Tensor:
    """Build a symmetric adjacency matrix for the SMPL-X skeleton.

    Args:
        num_joints: Total number of joints (default 148 for SMPL-X).

    Returns:
        A float tensor of shape ``(num_joints, num_joints)`` with 1.0 at
        connected joint pairs and 0.0 elsewhere.
    """
    A = torch.zeros(num_joints, num_joints, dtype=torch.float32)

    all_edges = _BODY_EDGES + _LEFT_HAND_EDGES + _RIGHT_HAND_EDGES + _build_face_edges()
    for i, j in all_edges:
        if i < num_joints and j < num_joints:
            A[i, j] = 1.0
            A[j, i] = 1.0

    return A


def _symmetric_normalise(A: torch.Tensor) -> torch.Tensor:
    """Apply symmetric normalisation: D^{-1/2} A D^{-1/2}.

    This implements c_{ij} = sqrt(|N(i)|) * sqrt(|N(j)|) normalisation
    used in Kipf & Welling (2017).

    Args:
        A: Adjacency matrix of shape ``(V, V)``.

    Returns:
        Normalised adjacency ``A_hat`` of shape ``(V, V)``.
    """
    # Add self-loops
    A_hat = A + torch.eye(A.size(0), dtype=A.dtype, device=A.device)
    # Degree vector
    deg = A_hat.sum(dim=1).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    # D^{-1/2} A_hat D^{-1/2}
    norm_A = deg_inv_sqrt.unsqueeze(1) * A_hat * deg_inv_sqrt.unsqueeze(0)
    return norm_A


# ---------------------------------------------------------------------------
# Graph Convolution Layer
# ---------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """Single graph convolution layer with symmetric normalisation.

    Implements:
        f_i^{(l+1)} = sigma( sum_{j in N(i)} (1/c_{ij}) W^{(l)} f_j^{(l)} + b^{(l)} )

    where c_{ij} = sqrt(|N(i)|) * sqrt(|N(j)|).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(0)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features of shape ``(..., V, C_in)``.
            norm_adj: Pre-normalised adjacency ``(V, V)``.

        Returns:
            Updated node features ``(..., V, C_out)``.
        """
        # x: (B, T, V, C_in)  |  norm_adj: (V, V)
        # Linear transform: (B, T, V, C_out)
        support = torch.matmul(x, self.weight)
        # Graph diffusion: norm_adj @ support  → aggregate neighbours
        # norm_adj is (V, V), support is (B, T, V, C_out)
        # We use einsum for clarity: 'vw,btwc->btwc'
        output = torch.einsum("vw,btwc->btwc", norm_adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


# ---------------------------------------------------------------------------
# Full GCN Stream
# ---------------------------------------------------------------------------

class PoseGCNStream(nn.Module):
    """Graph Convolutional Network stream for body-pose evaluation.

    Architecture:
        4 GCN layers with channel progression [3, 64, 128, 128],
        each followed by BatchNorm, ReLU, and Dropout(p=0.1).
        Global average pooling over joints **and** frames yields
        f_pose in R^{128}.

    Args:
        num_joints: Number of skeleton joints (default 148 for SMPL-X).
        dropout: Dropout probability after each GCN layer.
        pretrained_adj: Optional pre-built adjacency matrix.  When *None*,
            ``build_adjacency_matrix()`` is called at construction time.
    """

    def __init__(
        self,
        num_joints: int = 148,
        dropout: float = 0.1,
        pretrained_adj: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints

        # Build / register adjacency matrix as a buffer (not a parameter)
        if pretrained_adj is not None:
            adj = pretrained_adj.float()
        else:
            adj = build_adjacency_matrix(num_joints)
        norm_adj = _symmetric_normalise(adj)
        self.register_buffer("norm_adj", norm_adj)

        # GCN layer channel progression
        channels = [3, 64, 128, 128]
        self.gcn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

        for i in range(len(channels) - 1):
            self.gcn_layers.append(GraphConvLayer(channels[i], channels[i + 1]))
            self.bn_layers.append(nn.BatchNorm1d(channels[i + 1]))

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GCN stream.

        Args:
            keypoints: 3-D keypoints of shape ``(B, T, V, 3)`` where
                *B* = batch, *T* = frames, *V* = 148 joints.

        Returns:
            Pose feature vector ``f_pose`` of shape ``(B, 128)``.
        """
        x = keypoints  # (B, T, V, 3)
        for gcn, bn in zip(self.gcn_layers, self.bn_layers):
            x = gcn(x, self.norm_adj)          # (B, T, V, C)
            # BatchNorm expects (N, C) — reshape for 4-D input
            B, T, V, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, T, V)
            x = bn(x.view(B * T, C, V))              # BN over channel dim
            x = x.view(B, C, T, V).permute(0, 2, 3, 1).contiguous()  # back to (B, T, V, C)
            x = F.relu(x)
            x = self.dropout(x)

        # Global average pooling over joints (dim=2) and frames (dim=1)
        f_pose = x.mean(dim=2).mean(dim=1)  # (B, 128)
        return f_pose
