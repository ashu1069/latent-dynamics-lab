# Recurrent State-Space Model (RSSM)

A minimal, educational implementation of an RSSM for learning latent dynamics from pixel observations and actions. Designed for understanding and collaboration.

## Overview

The RSSM learns a compact latent representation of a dynamical system from high-dimensional observations (images). It can:

- **Reconstruct** observations from the latent state
- **Predict** future frames in open-loop (conditioned on past observations and actions)
- Handle **partial observability** (e.g., occlusion) via the stochastic latent

### Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                      At each timestep t                   │
                    └─────────────────────────────────────────────────────────┘

  obs_t ──► Encoder ──► embed_t
                            │
                            ▼
  (h, a_t) ──────────► Prior ─────► z_prior ~ N(μ_prior, σ_prior)   [predict without obs]
                            │
  (h, embed_t) ─────► Posterior ─► z ~ N(μ_post, σ_post)          [infer from obs]
                            │
                            ▼
  (h, z) ───────────► Decoder ───► recon_t
                            │
  concat(z, a_t), h ─► GRU ──────► h'
```

- **h**: Deterministic recurrent state (GRU hidden state)
- **z**: Stochastic latent (Gaussian)
- **Prior**: Predicts z from (h, a) — used for open-loop rollout
- **Posterior**: Infers z from (h, embed) — used during training when obs is available

## Setup

```bash
cd "reccurent state space models"
uv sync   # or: pip install -e .
```

Requires Python ≥3.10, PyTorch, NumPy, Matplotlib.

## Usage

### Training

```bash
python train.py --epochs 100 --batch_size 32 --seq_len 64
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--batch_size` | 32 | Batch size |
| `--seq_len` | 64 | Sequence length |
| `--grid_size` | 32 | Moving-dot grid size (H×W) |
| `--epochs` | 100 | Training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--use_occlusion` | True | Enable occlusion (partial observability) |
| `--no_occlusion` | - | Disable occlusion |
| `--output_dir` | outputs | Where to save checkpoints and plots |
| `--viz_interval` | 10 | Plot every N epochs |
| `--K` | 8 | Frames to condition on for open-loop evaluation |

### Outputs

- `outputs/model.pt` — Trained model weights
- `outputs/recon_epoch*.png` — Input vs reconstruction
- `outputs/rollout_epoch*.png` — True vs open-loop rollout
- `outputs/loss_curves.png` — Training loss
- `outputs/eval_rollout.png` — Final evaluation rollout

## Data: Moving Dot

The dataset generates sequences of a dot moving on a 2D grid with optional occlusion:

- **Observations**: `(B, T, 1, H, W)` — grayscale images
- **Actions**: `(B, T, 2)` — (ax, ay) velocity control
- **States**: `(B, T, 4)` — ground-truth (x, y, vx, vy)

The dot bounces off boundaries (reflect) and can be occluded by random rectangles to induce partial observability.

## Model Components

### `model.py`

| Class | Role |
|-------|------|
| `RSSMConfig` | Hyperparameters (dims, CNN architecture) |
| `CNNEncoder` | obs → embed |
| `CNNDecoder` | (h, z) → recon |
| `RSSM` | Full model: prior, posterior, GRU, encode/decode |

Key methods:

- `RSSM.forward(obs, actions, use_posterior=True)` — Full sequence forward (training)
- `RSSM.open_loop_rollout(obs_context, actions_context, actions_future, K)` — Condition on K frames, then predict with prior only

### `dataset.py`

- `MovingDotConfig` — Data generation settings
- `generate_moving_dot_sequences()` — Batch generator
- `MovingDotDataset` — PyTorch Dataset wrapper
- `collate_moving_dot()` — Collate for DataLoader

### `utils.py`

- `set_seed()` — Reproducibility
- `kl_balance()` — KL balancing to prevent posterior collapse

## Training Objective

```
L = L_recon + β · L_KL_balanced
```

- **L_recon**: MSE between observations and reconstructions
- **L_KL**: KL(posterior ‖ prior) — encourages latent to be informative while staying close to prior
- **KL balancing**: `balance * kl + (1 - balance) * kl.detach()` — stabilizes training
- **Free bits**: `kl.clamp(min=free_bits)` — prevents posterior collapse
