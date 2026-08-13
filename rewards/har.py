"""
signrl_diff.rewards.har
~~~~~~~~~~~~~~~~~~~~~~~

Hierarchical Articulation-Aware Reward (HAR) for the SignRL-Diff pipeline.

HAR decomposes the terminal reward into intermediate per-step rewards
during the denoising MDP.  At each diffusion step *k*:

1. **Partial decode** the noisy latent ``z_k`` via the VAE decoder to
   obtain a noisy video approximation ``x_tilde_k``.
2. **Estimate 3D keypoints** from ``x_tilde_k`` (simulated FrankMocap).
3. **Compute sub-rewards** using the frozen SRN's individual streams:
   - ``r_body``: GCN stream on estimated keypoints (pose quality)
   - ``r_hand``: Hand stream on cropped hand regions (articulation)
   - ``r_expr``: Semantic stream on CLIP features (expression/alignment)
4. **Dynamic weighting** via ``MLP_gate(text_condition)`` which outputs
   softmax weights ``[w_body, w_hand, w_expr]``.
5. **Discounted combination**:
   ``r_k = gamma^{K-k} * (w_body * r_body + w_hand * r_hand + w_expr * r_expr)``

References
----------
* Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
* SignRL-Diff: RL-based fine-tuning of video diffusion for sign language.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalReward(nn.Module):
    """Hierarchical Articulation-Aware Reward (HAR) system.

    The HAR module wraps a frozen Sub-Reward Network and a VAE decoder
    to produce intermediate rewards at every denoising step.  It adds
    a learnable gating MLP (``MLP_gate``) that produces dynamic weights
    for the three sub-reward streams conditioned on the text prompt.

    Parameters
    ----------
    srn : nn.Module
        A frozen :class:`~signrl_diff.models.srn.SubRewardNetwork`
        instance.  Its individual streams (``gcn_stream``,
        ``hand_stream``, ``semantic_stream``) are used to compute
        sub-rewards.
    vae_decoder : nn.Module
        The decoder portion of the video VAE (an
        :class:`~signrl_diff.models.diffusion.vae.AutoencoderKL` or
        its ``.decoder`` attribute).  Used to partially decode noisy
        latents into approximate video frames.
    text_dim : int
        Dimensionality of the text condition embedding fed to the
        gating MLP (default 1024).
    gate_hidden_dim : int
        Hidden dimensionality of the gating MLP (default 256).
    gamma : float
        Discount factor applied as ``gamma^{K-k}`` for intermediate
        rewards (default 0.99).
    """

    def __init__(
        self,
        srn: nn.Module,
        vae_decoder: nn.Module,
        text_dim: int = 1024,
        gate_hidden_dim: int = 256,
        gamma: float = 0.99,
    ) -> None:
        super().__init__()

        self.srn = srn
        self.vae_decoder = vae_decoder
        self.gamma: float = gamma

        # Freeze the SRN -- its weights are pre-trained in Phase 2
        for param in self.srn.parameters():
            param.requires_grad = False
        self.srn.eval()

        # Freeze the VAE decoder -- pre-trained in Phase 1
        for param in self.vae_decoder.parameters():
            param.requires_grad = False
        self.vae_decoder.eval()

        # ------------------------------------------------------------------
        # MLP_gate: AvgPool(c over tokens) -> FC -> ReLU -> FC -> Softmax
        # Produces 3 weights: [w_body, w_hand, w_expr]
        # ------------------------------------------------------------------
        self.MLP_gate = nn.Sequential(
            nn.Linear(text_dim, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, 3),
            nn.Softmax(dim=-1),
        )

        # ------------------------------------------------------------------
        # Lightweight expression head: maps semantic alignment score to a
        # scalar sub-reward.  The SRN's semantic stream outputs a cosine
        # similarity in [-1, 1]; this head rescales it for the HAR.
        # ------------------------------------------------------------------
        self.expr_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        # ------------------------------------------------------------------
        # Sub-reward score heads: project stream features to scalar rewards
        # ------------------------------------------------------------------
        self.body_score_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.hand_score_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    # ------------------------------------------------------------------
    # Gating weight computation
    # ------------------------------------------------------------------

    def _compute_gate_weights(
        self, text_condition: torch.Tensor
    ) -> torch.Tensor:
        """Compute dynamic sub-reward weights from the text condition.

        The text condition ``c`` of shape ``(B, L, D_text)`` is
        average-pooled over the token dimension *L* to produce a
        single vector, then passed through the gating MLP.

        Parameters
        ----------
        text_condition : Tensor, shape ``(B, L, D_text)``
            Text embedding tensor.

        Returns
        -------
        weights : Tensor, shape ``(B, 3)``
            Softmax weights ``[w_body, w_hand, w_expr]`` summing to 1.
        """
        # Average pool over token dimension
        c_pooled = text_condition.mean(dim=1)  # (B, D_text)
        weights = self.MLP_gate(c_pooled)       # (B, 3)
        return weights

    # ------------------------------------------------------------------
    # Partial decode helper
    # ------------------------------------------------------------------

    def _partial_decode(self, z_k: torch.Tensor) -> torch.Tensor:
        """Decode a noisy latent into an approximate video.

        The latent ``z_k`` of shape ``(B, T, C, H, W)`` is reshaped
        to ``(B*T, C, H, W)`` for the frame-wise VAE decoder, then
        reshaped back to ``(B, T, 3, H_out, W_out)``.

        Parameters
        ----------
        z_k : Tensor, shape ``(B, T, C, H, W)``
            Noisy latent at diffusion step *k*.

        Returns
        -------
        x_tilde : Tensor, shape ``(B, T, 3, H_out, W_out)``
            Approximate (noisy) video reconstruction.
        """
        B, T, C, Hl, Wl = z_k.shape

        # Check if the decoder is a full AutoencoderKL (has .decode method)
        # or just the Decoder module
        if hasattr(self.vae_decoder, 'decode'):
            x_tilde = self.vae_decoder.decode(z_k)
        else:
            z_flat = z_k.reshape(B * T, C, Hl, Wl)
            x_flat = self.vae_decoder(z_flat)   # (B*T, 3, H_out, W_out)
            _, C_out, H_out, W_out = x_flat.shape
            x_tilde = x_flat.reshape(B, T, C_out, H_out, W_out)

        return x_tilde

    # ------------------------------------------------------------------
    # Keypoint estimation (simulated FrankMocap)
    # ------------------------------------------------------------------

    def _estimate_keypoints(
        self, video: torch.Tensor
    ) -> torch.Tensor:
        """Estimate 3D keypoints from video frames.

        In the full pipeline this would call FrankMocap externally.
        Here we use a lightweight differentiable proxy: the SRN's GCN
        stream is run with pseudo-random keypoints derived from the
        video's spatial statistics, providing a gradient path through
        the reward.

        For production use, replace this with actual FrankMocap
        inference (non-differentiable) or a learned keypoint estimator.

        Parameters
        ----------
        video : Tensor, shape ``(B, T, 3, H, W)``
            Approximate video frames.

        Returns
        -------
        keypoints : Tensor, shape ``(B, T, 148, 3)``
            Estimated 3D SMPL-X keypoints.
        """
        B, T, C, H, W = video.shape
        num_joints = 148

        # Use spatial mean/std of video frames as a proxy for joint
        # positions.  This creates a differentiable path from the
        # decoded video to the keypoint-based sub-rewards.
        frame_feats = video.mean(dim=2)           # (B, T, H, W)

        # Compute spatial statistics per frame
        h_coords = torch.linspace(-1, 1, H, device=video.device)
        w_coords = torch.linspace(-1, 1, W, device=video.device)
        h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing="ij")

        # Weighted spatial coordinates (centre of mass proxy)
        weights = F.softmax(frame_feats.reshape(B, T, -1), dim=-1)
        weights_2d = weights.reshape(B, T, H, W)

        cx = (weights_2d * w_grid.unsqueeze(0).unsqueeze(0)).sum(dim=[2, 3])
        cy = (weights_2d * h_grid.unsqueeze(0).unsqueeze(0)).sum(dim=[2, 3])

        # Build pseudo-keypoints: replicate centre of mass with joint offsets
        # Joint offsets are fixed skeleton positions (deterministic)
        joint_offsets_x = torch.linspace(-0.5, 0.5, num_joints, device=video.device)
        joint_offsets_y = torch.linspace(-0.8, 0.8, num_joints, device=video.device)
        joint_offsets_z = torch.zeros(num_joints, device=video.device)

        # Broadcast: (B, T, 1) + (num_joints,) -> (B, T, num_joints)
        kpt_x = cx.unsqueeze(-1) + joint_offsets_x.unsqueeze(0).unsqueeze(0) * 0.3
        kpt_y = cy.unsqueeze(-1) + joint_offsets_y.unsqueeze(0).unsqueeze(0) * 0.3
        kpt_z = joint_offsets_z.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

        keypoints = torch.stack([kpt_x, kpt_y, kpt_z], dim=-1)  # (B, T, 148, 3)
        return keypoints

    # ------------------------------------------------------------------
    # Intermediate reward computation
    # ------------------------------------------------------------------

    def compute_intermediate_reward(
        self,
        z_k: torch.Tensor,
        text_condition: torch.Tensor,
        k: int,
        K: int,
    ) -> torch.Tensor:
        """Compute the intermediate per-step reward at denoising step *k*.

        Steps:
        1. Partial decode ``z_k`` to get noisy video ``x_tilde_k``.
        2. Estimate keypoints from ``x_tilde_k``.
        3. Run SRN sub-reward streams on the approximate video.
        4. Compute dynamic weights from ``MLP_gate(text_condition)``.
        5. Combine with discount factor: ``r_k = gamma^{K-k} * weighted_sum``.

        Parameters
        ----------
        z_k : Tensor, shape ``(B, T, C, H, W)``
            Noisy latent at step *k*.
        text_condition : Tensor, shape ``(B, L, D_text)``
            Text embedding.
        k : int
            Current diffusion step (0 = clean, K = pure noise).
        K : int
            Total number of diffusion steps.

        Returns
        -------
        r_k : Tensor, shape ``(B,)``
            Intermediate reward at step *k*.
        """
        with torch.no_grad():
            # Step 1: Partial decode
            x_tilde = self._partial_decode(z_k)  # (B, T, 3, H, W)

            # Step 2: Estimate keypoints
            keypoints = self._estimate_keypoints(x_tilde)  # (B, T, 148, 3)

        # Step 3: Run SRN sub-reward streams (SRN is frozen, but we
        # want gradients through the gate and score heads)
        with torch.no_grad():
            # Extract sub-reward features from the SRN
            # Use CLIP text embedding (first 512 dims or project)
            text_emb_512 = text_condition.mean(dim=1)[:, :512]
            if text_emb_512.shape[-1] < 512:
                text_emb_512 = F.pad(text_emb_512, (0, 512 - text_emb_512.shape[-1]))

            sub_rewards = self.srn.compute_sub_rewards(
                video=x_tilde,
                keypoints=keypoints,
                text_emb=text_emb_512,
                video_emb=None,
                hand_crops=None,
            )

        # Extract individual stream outputs
        f_pose = sub_rewards["f_pose"]       # (B, 128)
        f_hand = sub_rewards["f_hand"]       # (B, 128)
        f_sem = sub_rewards["f_sem"]          # (B,)

        # Compute scalar sub-rewards through learnable heads
        r_body = self.body_score_head(f_pose).squeeze(-1)    # (B,)
        r_hand = self.hand_score_head(f_hand).squeeze(-1)    # (B,)
        r_expr = self.expr_head(f_sem.unsqueeze(-1)).squeeze(-1)  # (B,)

        # Step 4: Dynamic weights from gating MLP
        weights = self._compute_gate_weights(text_condition)  # (B, 3)
        w_body = weights[:, 0]  # (B,)
        w_hand = weights[:, 1]  # (B,)
        w_expr = weights[:, 2]  # (B,)

        # Step 5: Discounted combination
        discount = self.gamma ** (K - k)
        r_k = discount * (w_body * r_body + w_hand * r_hand + w_expr * r_expr)

        return r_k

    # ------------------------------------------------------------------
    # Terminal reward computation
    # ------------------------------------------------------------------

    def compute_terminal_reward(
        self,
        z_0: torch.Tensor,
        text_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the terminal reward for a fully denoised latent.

        At ``k = 0``, the latent ``z_0`` is decoded to a clean video,
        and the full SRN evaluation is performed (all four streams
        fused through the SRN's fusion MLP).

        Parameters
        ----------
        z_0 : Tensor, shape ``(B, T, C, H, W)``
            Final (clean) latent after all denoising steps.
        text_condition : Tensor, shape ``(B, L, D_text)``
            Text embedding.

        Returns
        -------
        r_0 : Tensor, shape ``(B,)``
            Terminal reward in [-1, 1].
        """
        with torch.no_grad():
            # Decode clean latent to video
            v = self._partial_decode(z_0)  # (B, T, 3, H, W)

            # Estimate keypoints from clean video
            keypoints = self._estimate_keypoints(v)  # (B, T, 148, 3)

            # Prepare CLIP text embedding (512-d)
            text_emb_512 = text_condition.mean(dim=1)[:, :512]
            if text_emb_512.shape[-1] < 512:
                text_emb_512 = F.pad(text_emb_512, (0, 512 - text_emb_512.shape[-1]))

            # Full SRN evaluation (all streams fused)
            r_0 = self.srn(
                video=v,
                keypoints=keypoints,
                text_emb=text_emb_512,
                video_emb=None,
                hand_crops=None,
            )  # (B,) in [-1, 1]

        return r_0

    # ------------------------------------------------------------------
    # Convenience: full forward selects terminal or intermediate
    # ------------------------------------------------------------------

    def forward(
        self,
        z_k: torch.Tensor,
        text_condition: torch.Tensor,
        k: int,
        K: int,
    ) -> torch.Tensor:
        """Unified interface: dispatches to terminal or intermediate reward.

        Parameters
        ----------
        z_k : Tensor, shape ``(B, T, C, H, W)``
        text_condition : Tensor, shape ``(B, L, D_text)``
        k : int
            Current diffusion step.
        K : int
            Total diffusion steps.

        Returns
        -------
        reward : Tensor, shape ``(B,)``
        """
        if k == 0:
            return self.compute_terminal_reward(z_k, text_condition)
        return self.compute_intermediate_reward(z_k, text_condition, k, K)
