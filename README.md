# SignRL-Diff: RL-Guided Stable Diffusion for Sign Language Video Generation

SignRL-Diff is a three-phase training pipeline that produces high-quality sign language videos from text prompts. It combines a video diffusion model with reinforcement learning, using a hierarchical reward system that evaluates body pose quality, temporal coherence, semantic alignment, and hand articulation.

## Three-Phase Training Pipeline

### Phase 1: Diffusion Pre-training
Pre-trains the VideoUNet and AutoencoderKL on sign language datasets. The VAE encoder maps video clips to latent space; the UNet learns to predict Gaussian noise at random diffusion timesteps.

- **Objective**: L = E[||eps - UNet(z_t, t, c)||^2]
- **Optimizer**: AdamW, lr=1e-4, cosine annealing with warmup
- **EMA**: Decay 0.9999
- **Duration**: 500K steps

### Phase 2: Sub-Reward Network Training
Trains the SRN to evaluate generated videos along four axes. The frozen diffusion model generates K=8 candidates per sentence; quality metrics (MPJPE, temporal smoothness, CLIP similarity) establish rankings; the SRN learns to reproduce them.

- **Objective**: L = -E[log(exp(r_i) / sum_j exp(r_j))]
- **Optimizer**: Adam, lr=3e-4
- **Duration**: 50K steps

### Phase 3: RL Fine-tuning with PPO
Fine-tunes the diffusion model using Proximal Policy Optimization. LoRA adapters (rank=16) are injected into UNet attention layers. The policy network outputs latent-space corrections at each denoising step.

- **Algorithm**: PPO with GAE (gamma=0.99, lambda=0.95)
- **Environments**: 4 parallel VecDenoisingEnv instances
- **Rollout**: 50 steps per collection
- **Update**: 4 PPO epochs, mini-batch size 64
- **Duration**: 200K steps



### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-compatible GPU recommended (4x GPUs for full training)

## Quick Start

### Data Preparation

Organize your datasets as follows:

```
data/
  PHOENIX14T/
    videos/
      sample_001.mp4
      ...
    annotations.json
    embeddings/
      sample_001.pt
  How2Sign/
    ...
  USTC-CSL/
    ...
```

Each `annotations.json` should contain:
```json
[
  {"video": "sample_001.mp4", "gloss": "HELLO WORLD", "split": "train"},
  ...
]
```

### Training

#### Phase 1: Diffusion Pre-training

```bash
python -m signrl_diff.scripts.train_phase1 \
    --config configs/default.yaml \
    --data_dir ./data \
    --output_dir ./checkpoints/phase1 \
    --seed 42
```

To resume training:
```bash
python -m signrl_diff.scripts.train_phase1 \
    --config configs/default.yaml \
    --data_dir ./data \
    --output_dir ./checkpoints/phase1 \
    --resume ./checkpoints/phase1/latest.pt
```

#### Phase 2: SRN Training

```bash
python -m signrl_diff.scripts.train_phase2 \
    --config configs/default.yaml \
    --data_dir ./data \
    --diffusion_checkpoint ./checkpoints/phase1/final.pt \
    --output_dir ./checkpoints/phase2
```

#### Phase 3: RL Fine-tuning with PPO

```bash
python -m signrl_diff.scripts.train_phase3 \
    --config configs/default.yaml \
    --phase1_ckpt ./checkpoints/phase1/final.pt \
    --phase2_ckpt ./checkpoints/phase2/final.pt \
    --output_dir ./checkpoints/phase3
```

### Inference

Generate a sign language video from text:

```bash
python -m signrl_diff.scripts.inference \
    --text "HELLO WORLD HOW ARE YOU" \
    --checkpoint_dir ./checkpoints \
    --output output.mp4 \
    --num_frames 32 \
    --denoising_steps 50 \
    --fps 16
```

Generate as GIF:
```bash
python -m signrl_diff.scripts.inference \
    --text "THANK YOU VERY MUCH" \
    --checkpoint_dir ./checkpoints \
    --output output.gif \
    --num_frames 24
```

Without policy corrections (baseline diffusion only):
```bash
python -m signrl_diff.scripts.inference \
    --text "GOOD MORNING" \
    --checkpoint_dir ./checkpoints \
    --output baseline.mp4 \
    --no_policy
```

## Model Architecture Details

### Denoising MDP

The reverse diffusion process is formulated as a Markov Decision Process:

| Component | Description |
|-----------|-------------|
| **State** s_k | (z_k, k, c, eps_hat_k) - noisy latent, timestep, text condition, UNet prediction |
| **Action** a_k | 129-d vector: [global(64), hand_left(32), hand_right(32), scale(1)] |
| **Transition** | z_{k-1} = scheduler.step(eps_hat_k + correction(a_k), k, z_k) |
| **Reward** | HAR: intermediate per-step + terminal at k=0 |

### VideoUNet

- **Architecture**: Encoder-decoder with skip connections
- **Channel config**: [128, 256, 512, 512] per resolution level
- **Attention**: Spatial self-attention, temporal self-attention, cross-attention over text
- **Conditioning**: Sinusoidal timestep embedding (512-d) + text cross-attention
- **LoRA**: Rank-16 low-rank adapters on all attention projections

### AutoencoderKL (VAE)

- **Encoder**: 3 downsampling stages, 8x spatial reduction (256x256 -> 32x32)
- **Decoder**: 3 upsampling stages, 8x spatial increase
- **Latent**: 4 channels, 32x32 spatial, reparameterization trick
- **Loss**: L1 reconstruction + KL divergence (weight=1e-4)

### Policy Network

- **Encoder**: Flatten [z_k; eps_hat_k] -> MLP(1024->512) + sinusoidal PE(k)
- **Cross-attention**: Single query from state, keys/values from text
- **Action head**: MLP(512->256->128->258) producing mean + log-std
- **Distribution**: Diagonal Gaussian over 129-d actions

### Value Network

- **Architecture**: Same encoder as Policy Network
- **Value head**: MLP(512->256->1) producing scalar V(s_k)

### Sub-Reward Network (SRN)

| Stream | Input | Output | Architecture |
|--------|-------|--------|-------------|
| **Pose GCN** | Keypoints (B,T,148,3) | f_pose (B,128) | 4 GCN layers with SMPL-X skeleton graph |
| **Temporal TCN** | Pose deltas (B,T-1,444) | f_temp (B,256) | 6 dilated causal conv layers (receptive field: 126 frames) |
| **Semantic** | CLIP text + video emb | f_sem (B,) | Linear projection + cosine similarity |
| **Hand CNN** | Hand crops (B,T,2,3,128,128) | f_hand (B,128) | EfficientNet-B4 backbone + MLP |

Fusion: concat(513-d) -> MLP(513->256->128->1) -> Tanh -> reward in [-1, 1]

### Hierarchical Articulation-Aware Reward (HAR)

- **Intermediate rewards**: Per-step evaluation during denoising
- **Sub-rewards**: r_body (GCN), r_hand (CNN), r_expr (semantic)
- **Dynamic gating**: MLP_gate(text_condition) -> softmax weights [w_body, w_hand, w_expr]
- **Discounted**: r_k = gamma^(K-k) * weighted_sum
- **Terminal**: Full SRN evaluation at k=0

### DDPMScheduler

- **Forward process**: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
- **Reverse process**: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_hat) + sigma_t * eps
- **Beta schedule**: Linear (1e-4 to 0.02) over 1000 steps
- **Noise prediction**: Epsilon-parameterized

### LoRA (Low-Rank Adaptation)

- Applied to all attention projections (Q, K, V, O) in the UNet
- y = W_frozen(x) + (alpha/r) * B(A(x))
- A: (in_features, r), kaiming-uniform init
- B: (r, out_features), zero init (no contribution at start)
- Rank r=16, alpha=16.0, effective scale=1.0

## Datasets

| Dataset | Language | Type | Size |
|---------|----------|------|------|
| **PHOENIX14T** | German (DGS) | Continuous signing | ~7K sentences |
| **How2Sign** | American (ASL) | Instructional content | ~35K sentences |
| **USTC-CSL** | Chinese (CSL) | Isolated & continuous | ~5K sentences |

Each dataset provides:
- RGB video clips (256x256 resolution)
- Gloss-level text annotations
- Pre-computed CLIP text embeddings (1024-d)

## Configuration

All hyperparameters are defined in `configs/default.yaml`. Key settings:

```yaml
model:
  unet_channels: [128, 256, 512, 512]
  lora_rank: 16
  num_frames: 16
  text_dim: 1024

rl:
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  lr: 3.0e-5
  num_envs: 4
  rollout_steps: 50
  total_steps: 200000

training:
  phase1_steps: 500000
  phase2_steps: 50000
  phase3_steps: 200000
  batch_size: 4
```

## Citation

```bibtex
@article{signrl_diff2024,
  title   = {SignRL-Diff: RL-Guided Stable Diffusion for Sign Language Video Generation},
  author  = {SignRL-Diff Contributors},
  journal = {arXiv preprint},
  year    = {2024},
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

```
MIT License

Copyright (c) 2024 SignRL-Diff Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
