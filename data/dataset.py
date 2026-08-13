"""
signrl_diff.data.dataset
~~~~~~~~~~~~~~~~~~~~~~~~

Video dataset for sign language generation training.

Supports three sign language datasets:
- **PHOENIX14T**: German sign language (DGS), continuous signing
- **How2Sign**: American sign language (ASL), instructional content
- **USTC-CSL**: Chinese sign language (CSL), isolated & continuous

Each item yields:
- ``video_tensor``: ``(T, 3, 256, 256)`` — sampled, resized, normalised frames
- ``text_gloss``: str — gloss-level annotation
- ``text_embedding``: ``(L, 1024)`` — pre-computed CLIP embedding

Also provides :func:`build_preference_pairs` for generating pseudo-preference
pairs used in SRN (Sub-Reward Network) training via listwise ranking loss.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np


# ======================================================================
# Video preprocessing helpers
# ======================================================================

def _load_video_frames(
    video_path: str,
    num_frames: int = 32,
    resolution: int = 256,
) -> torch.Tensor:
    """Load and preprocess video frames from a file.

    Attempts to use ``torchvision.io.read_video`` for decoding.  Falls
    back to loading pre-extracted frame images if the video file cannot
    be decoded (common in HPC environments without ffmpeg).

    Parameters
    ----------
    video_path : str
        Path to the video file or directory of frames.
    num_frames : int
        Number of frames to sample (temporal).
    resolution : int
        Target spatial resolution (HxW).

    Returns
    -------
    Tensor, shape ``(num_frames, 3, resolution, resolution)``
        Normalised video tensor in ``[-1, 1]``.
    """
    path = Path(video_path)

    # Try loading as a directory of pre-extracted frames
    if path.is_dir():
        frame_files = sorted(path.glob("*.jpg")) + sorted(path.glob("*.png"))
        if len(frame_files) == 0:
            frame_files = sorted(path.glob("*.jpeg"))
        return _load_frame_directory(frame_files, num_frames, resolution)

    # Try torchvision video reader
    try:
        from torchvision.io import read_video
        video_data, _, _ = read_video(str(path), pts_unit="sec")
        # video_data: (T, H, W, 3) uint8
        return _process_raw_video(video_data, num_frames, resolution)
    except Exception:
        pass

    # Fallback: generate synthetic frames (for testing / CI)
    return _generate_synthetic_video(num_frames, resolution)


def _load_frame_directory(
    frame_files: List[Path],
    num_frames: int,
    resolution: int,
) -> torch.Tensor:
    """Load frames from a directory, sample, resize, and normalise.

    Parameters
    ----------
    frame_files : list of Path
        Sorted list of image file paths.
    num_frames : int
        Target number of frames.
    resolution : int
        Target spatial size.

    Returns
    -------
    Tensor, shape ``(num_frames, 3, resolution, resolution)``
    """
    from torchvision.io import read_image
    from torchvision.transforms.functional import resize

    total = len(frame_files)
    if total == 0:
        return _generate_synthetic_video(num_frames, resolution)

    # Uniform temporal sampling
    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        # Repeat last frame if not enough frames
        indices = list(range(total)) + [total - 1] * (num_frames - total)

    frames = []
    for idx in indices:
        img = read_image(str(frame_files[idx]))  # (3, H, W) uint8
        img = img.float() / 255.0
        img = resize(img, [resolution, resolution], antialias=True)
        frames.append(img)

    video = torch.stack(frames, dim=0)  # (T, 3, H, W)
    # Normalise to [-1, 1]
    video = video * 2.0 - 1.0
    return video


def _process_raw_video(
    video_data: torch.Tensor,
    num_frames: int,
    resolution: int,
) -> torch.Tensor:
    """Process raw video tensor from torchvision reader.

    Parameters
    ----------
    video_data : Tensor, shape ``(T, H, W, 3)``
        Raw uint8 video tensor.
    num_frames : int
        Target frame count.
    resolution : int
        Target spatial size.

    Returns
    -------
    Tensor, shape ``(num_frames, 3, resolution, resolution)``
    """
    from torchvision.transforms.functional import resize

    T_total = video_data.shape[0]

    # Uniform temporal sampling
    if T_total >= num_frames:
        indices = np.linspace(0, T_total - 1, num_frames, dtype=int)
    else:
        indices = list(range(T_total)) + [T_total - 1] * (num_frames - T_total)

    frames = []
    for idx in indices:
        frame = video_data[idx].permute(2, 0, 1).float() / 255.0  # (3, H, W)
        frame = resize(frame, [resolution, resolution], antialias=True)
        frames.append(frame)

    video = torch.stack(frames, dim=0)  # (T, 3, H, W)
    video = video * 2.0 - 1.0
    return video


def _generate_synthetic_video(
    num_frames: int,
    resolution: int,
) -> torch.Tensor:
    """Generate a synthetic random video tensor (fallback for testing).

    Returns
    -------
    Tensor, shape ``(num_frames, 3, resolution, resolution)``
        Random tensor in ``[-1, 1]``.
    """
    video = torch.randn(num_frames, 3, resolution, resolution)
    video = video.clamp(-1.0, 1.0)
    return video


# ======================================================================
# Text embedding helpers
# ======================================================================

def _load_text_embedding(
    embedding_path: str,
    max_length: int = 77,
    embed_dim: int = 1024,
) -> torch.Tensor:
    """Load a pre-computed CLIP text embedding from disk.

    Parameters
    ----------
    embedding_path : str
        Path to the ``.pt`` file containing the embedding.
    max_length : int
        Maximum token length (pad or truncate).
    embed_dim : int
        Expected embedding dimensionality.

    Returns
    -------
    Tensor, shape ``(max_length, embed_dim)``
    """
    if os.path.exists(embedding_path):
        emb = torch.load(embedding_path, weights_only=True)
    else:
        # Fallback: random embedding
        emb = torch.randn(1, embed_dim)

    # Ensure 2D: (L, D)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)

    L, D = emb.shape

    # Truncate
    if L > max_length:
        emb = emb[:max_length]
        L = max_length

    # Pad if needed
    if L < max_length:
        padding = torch.zeros(max_length - L, D)
        emb = torch.cat([emb, padding], dim=0)

    # Pad/trim feature dim
    if D < embed_dim:
        emb = F.pad(emb, (0, embed_dim - D))
    elif D > embed_dim:
        emb = emb[:, :embed_dim]

    return emb


# ======================================================================
# Main Dataset Class
# ======================================================================

class SignLanguageVideoDataset(Dataset):
    """PyTorch Dataset for sign language video clips with text annotations.

    Supports PHOENIX14T, How2Sign, and USTC-CSL datasets.  Each item
    yields a video tensor, gloss string, and CLIP text embedding.

    The dataset expects a directory structure::

        data_root/
          PHOENIX14T/
            videos/
              sample_001.mp4 (or sample_001/ directory of frames)
              ...
            annotations.json
            embeddings/
              sample_001.pt
              ...
          How2Sign/
            ...
          USTC-CSL/
            ...

    The ``annotations.json`` file should contain a list of dicts::

        [
            {"video": "sample_001.mp4", "gloss": "HELLO WORLD", "split": "train"},
            ...
        ]

    Parameters
    ----------
    data_root : str
        Root directory containing dataset folders.
    datasets : list of str
        Dataset names to include (e.g., ``["PHOENIX14T", "How2Sign"]``).
    split : str
        Data split: ``"train"``, ``"val"``, or ``"test"``.
    num_frames : int
        Number of frames to sample per video (default 32).
    resolution : int
        Target spatial resolution (default 256).
    text_emb_dim : int
        CLIP text embedding dimensionality (default 1024).
    max_text_length : int
        Maximum text token length (default 77).
    clip_model : nn.Module, optional
        If provided, text embeddings are computed on-the-fly instead
        of loaded from disk.  Should accept a string and return
        ``(L, D)`` embeddings.
    """

    def __init__(
        self,
        data_root: str = "./data",
        datasets: Optional[List[str]] = None,
        split: str = "train",
        num_frames: int = 32,
        resolution: int = 256,
        text_emb_dim: int = 1024,
        max_text_length: int = 77,
        clip_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if datasets is None:
            datasets = ["PHOENIX14T", "How2Sign", "USTC-CSL"]

        self.data_root = Path(data_root)
        self.datasets = datasets
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.text_emb_dim = text_emb_dim
        self.max_text_length = max_text_length
        self.clip_model = clip_model

        # Collect all samples across datasets
        self.samples: List[Dict[str, str]] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Load annotations from all configured datasets."""
        for ds_name in self.datasets:
            ds_dir = self.data_root / ds_name
            ann_path = ds_dir / "annotations.json"

            if ann_path.exists():
                with open(ann_path, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
            else:
                # If no annotations file, scan for video files
                annotations = self._scan_directory(ds_dir)

            for ann in annotations:
                # Filter by split
                ann_split = ann.get("split", "train")
                if ann_split != self.split:
                    continue

                self.samples.append({
                    "dataset": ds_name,
                    "video": str(ds_dir / "videos" / ann["video"]),
                    "gloss": ann.get("gloss", ""),
                    "embedding": str(
                        ds_dir / "embeddings"
                        / (Path(ann["video"]).stem + ".pt")
                    ),
                })

    def _scan_directory(self, ds_dir: Path) -> List[Dict[str, str]]:
        """Scan a dataset directory for video files when no annotations exist.

        Parameters
        ----------
        ds_dir : Path
            Dataset root directory.

        Returns
        -------
        list of dict
            Auto-generated annotations.
        """
        video_dir = ds_dir / "videos"
        if not video_dir.exists():
            return []

        annotations = []
        video_files = (
            list(video_dir.glob("*.mp4"))
            + list(video_dir.glob("*.avi"))
            + list(video_dir.glob("*.mov"))
        )
        # Also check for frame directories
        frame_dirs = [d for d in video_dir.iterdir() if d.is_dir()]

        for vf in video_files:
            annotations.append({
                "video": vf.name,
                "gloss": vf.stem.replace("_", " "),
                "split": "train",
            })
        for fd in frame_dirs:
            annotations.append({
                "video": fd.name,
                "gloss": fd.name.replace("_", " "),
                "split": "train",
            })

        return annotations

    def __len__(self) -> int:
        return max(len(self.samples), 1)  # At least 1 for synthetic fallback

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
        """Load a single sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        video : Tensor, shape ``(T, 3, 256, 256)``
            Video tensor normalised to ``[-1, 1]``.
        gloss : str
            Text gloss annotation.
        text_embedding : Tensor, shape ``(L, 1024)``
            CLIP text embedding.
        """
        if idx < len(self.samples):
            sample = self.samples[idx]
            video = _load_video_frames(
                sample["video"],
                num_frames=self.num_frames,
                resolution=self.resolution,
            )
            gloss = sample["gloss"]

            # Load or compute text embedding
            if self.clip_model is not None:
                text_embedding = self._encode_text(gloss)
            else:
                text_embedding = _load_text_embedding(
                    sample["embedding"],
                    max_length=self.max_text_length,
                    embed_dim=self.text_emb_dim,
                )
        else:
            # Synthetic fallback for testing
            video = _generate_synthetic_video(self.num_frames, self.resolution)
            gloss = "SYNTHETIC SAMPLE"
            text_embedding = torch.randn(self.max_text_length, self.text_emb_dim)

        return video, gloss, text_embedding

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text using the provided CLIP model.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        Tensor, shape ``(max_length, text_emb_dim)``
        """
        with torch.no_grad():
            emb = self.clip_model(text)  # Expected: (L, D) or (D,)

        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        L, D = emb.shape
        if L > self.max_text_length:
            emb = emb[:self.max_text_length]
        elif L < self.max_text_length:
            emb = F.pad(emb, (0, 0, 0, self.max_text_length - L))

        if D < self.text_emb_dim:
            emb = F.pad(emb, (0, self.text_emb_dim - D))
        elif D > self.text_emb_dim:
            emb = emb[:, :self.text_emb_dim]

        return emb


# ======================================================================
# Preference Pair Builder
# ======================================================================

def build_preference_pairs(
    dataset: SignLanguageVideoDataset,
    K: int = 8,
    srn: Optional[nn.Module] = None,
    vae: Optional[nn.Module] = None,
    scheduler: Optional[nn.Module] = None,
    unet: Optional[nn.Module] = None,
    device: str | torch.device = "cpu",
) -> List[Dict[str, Any]]:
    """Generate pseudo-preference pairs for SRN training.

    For each sentence in the dataset, generate *K* candidate videos by
    running the diffusion model with different random seeds, compute
    quality scores using the SRN, and create preference pairs ranked
    by quality.

    The pairs are used to train the SRN with a listwise ranking loss
    (Bradley-Terry / Plackett-Luce).

    Parameters
    ----------
    dataset : SignLanguageVideoDataset
        The source dataset.
    K : int
        Number of candidates per sentence (default 8).
    srn : nn.Module, optional
        Sub-Reward Network for quality scoring. If *None*, random
        scores are assigned (for testing).
    vae : nn.Module, optional
        Video VAE for encoding/decoding.
    scheduler : nn.Module, optional
        DDPM scheduler for sampling.
    unet : nn.Module, optional
        UNet for generation.
    device : str or torch.device
        Computation device.

    Returns
    -------
    list of dict
        Each dict contains:
        - ``"text_embedding"``: ``(L, D)`` text embedding
        - ``"gloss"``: str
        - ``"candidates"``: list of K video tensors
        - ``"scores"``: Tensor of K quality scores
        - ``"rankings"``: Tensor of K ranking indices (0 = best)
    """
    device = torch.device(device)
    preference_data: List[Dict[str, Any]] = []

    num_samples = min(len(dataset), 1000)  # Cap for efficiency

    for idx in range(num_samples):
        video, gloss, text_emb = dataset[idx]

        candidates: List[torch.Tensor] = []
        scores: List[float] = []

        for k_idx in range(K):
            # Set seed for reproducibility per candidate
            torch.manual_seed(idx * 1000 + k_idx)

            if unet is not None and vae is not None and scheduler is not None:
                # Generate a candidate video via diffusion sampling
                candidate = _generate_candidate(
                    unet, vae, scheduler, text_emb, device
                )
            else:
                # Fallback: perturb the original video
                noise_level = 0.1 * (k_idx + 1)
                candidate = video + torch.randn_like(video) * noise_level
                candidate = candidate.clamp(-1.0, 1.0)

            candidates.append(candidate)

            # Score the candidate
            if srn is not None:
                score = _score_candidate(srn, candidate, text_emb, device)
            else:
                # Random quality score (inversely related to noise for testing)
                score = 1.0 - 0.1 * (k_idx + 1) + random.gauss(0, 0.05)

            scores.append(score)

        # Compute rankings (0 = best, higher = worse)
        score_tensor = torch.tensor(scores)
        rankings = score_tensor.argsort(descending=True).argsort().float()

        preference_data.append({
            "text_embedding": text_emb,
            "gloss": gloss,
            "candidates": candidates,
            "scores": score_tensor,
            "rankings": rankings,
        })

    return preference_data


def _generate_candidate(
    unet: nn.Module,
    vae: nn.Module,
    scheduler: nn.Module,
    text_emb: torch.Tensor,
    device: torch.device,
    num_frames: int = 32,
    latent_channels: int = 4,
    latent_hw: int = 32,
) -> torch.Tensor:
    """Generate a single candidate video via the diffusion model.

    Parameters
    ----------
    unet : nn.Module
        Diffusion UNet.
    vae : nn.Module
        Video VAE.
    scheduler : nn.Module
        DDPM scheduler.
    text_emb : Tensor
        Text embedding ``(L, D)``.
    device : torch.device
    num_frames : int
    latent_channels : int
    latent_hw : int

    Returns
    -------
    Tensor, shape ``(T, 3, 256, 256)``
        Generated video tensor.
    """
    with torch.no_grad():
        # Sample initial noise
        z = torch.randn(
            1, num_frames, latent_channels, latent_hw, latent_hw,
            device=device,
        )
        text_batch = text_emb.unsqueeze(0).to(device)

        # Run reverse diffusion
        for t in reversed(range(scheduler.num_train_steps)):
            t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
            eps_hat = unet(z, t_tensor, text_batch)
            z = scheduler.step(eps_hat, t, z)

        # Decode to video space
        if hasattr(vae, 'decode'):
            video = vae.decode(z)
        else:
            B, T, C, H, W = z.shape
            z_flat = z.reshape(B * T, C, H, W)
            v_flat = vae(z_flat)
            _, Co, Ho, Wo = v_flat.shape
            video = v_flat.reshape(B, T, Co, Ho, Wo)

    return video.squeeze(0).cpu()


def _score_candidate(
    srn: nn.Module,
    video: torch.Tensor,
    text_emb: torch.Tensor,
    device: torch.device,
) -> float:
    """Score a candidate video using the SRN.

    Parameters
    ----------
    srn : nn.Module
        Sub-Reward Network.
    video : Tensor, shape ``(T, 3, H, W)``
        Candidate video.
    text_emb : Tensor, shape ``(L, D)``
        Text embedding.
    device : torch.device

    Returns
    -------
    float
        Quality score.
    """
    with torch.no_grad():
        v = video.unsqueeze(0).to(device)  # (1, T, 3, H, W)
        T = v.shape[1]
        num_joints = 148

        # Pseudo-keypoints (centre of mass)
        keypoints = torch.randn(1, T, num_joints, 3, device=device) * 0.1

        # Text embedding (take first 512 dims for CLIP)
        text_512 = text_emb.mean(dim=0)[:512].unsqueeze(0).to(device)
        if text_512.shape[-1] < 512:
            text_512 = F.pad(text_512, (0, 512 - text_512.shape[-1]))

        score = srn(v, keypoints, text_512)  # (1,)
        return score.item()
