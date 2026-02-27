"""
Moving-dot data generator with occlusion and control.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MovingDotConfig:
    """Configuration for the moving-dot environment."""
    grid_size: int = 32
    seq_len: int = 64
    dot_radius: float = 1.5
    dot_intensity: float = 1.0
    background: float = 0.0
    # Occlusion
    use_occlusion: bool = True
    occlusion_prob: float = 0.3
    occlusion_min_frames: int = 3
    occlusion_max_frames: int = 10
    occlusion_min_size: int = 4
    occlusion_max_size: int = 12
    # Dynamics
    max_velocity: float = 2.0
    action_scale: float = 0.5
    boundary: str = "reflect"  # "reflect" or "clip"
    seed: Optional[int] = None


def _reflect_boundary(pos: torch.Tensor, vel: torch.Tensor, grid_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reflect position and velocity at boundaries."""
    pos = pos.clone()
    vel = vel.clone()
    # Left/right (x)
    mask_left = pos[..., 0] < 0
    pos[..., 0][mask_left] = -pos[..., 0][mask_left]
    vel[..., 0][mask_left] = -vel[..., 0][mask_left]
    mask_right = pos[..., 0] >= grid_size
    pos[..., 0][mask_right] = 2 * grid_size - 1 - pos[..., 0][mask_right]
    vel[..., 0][mask_right] = -vel[..., 0][mask_right]
    # Top/bottom (y)
    mask_bottom = pos[..., 1] < 0
    pos[..., 1][mask_bottom] = -pos[..., 1][mask_bottom]
    vel[..., 1][mask_bottom] = -vel[..., 1][mask_bottom]
    mask_top = pos[..., 1] >= grid_size
    pos[..., 1][mask_top] = 2 * grid_size - 1 - pos[..., 1][mask_top]
    vel[..., 1][mask_top] = -vel[..., 1][mask_top]
    return pos, vel


def _clip_boundary(pos: torch.Tensor, vel: torch.Tensor, grid_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip position at boundaries and zero velocity."""
    pos = pos.clamp(0, grid_size - 1e-6)
    vel = torch.where(
        (pos[..., 0:1] <= 0) | (pos[..., 0:1] >= grid_size - 1),
        torch.zeros_like(vel),
        vel
    )
    vel = torch.where(
        (pos[..., 1:2] <= 0) | (pos[..., 1:2] >= grid_size - 1),
        torch.zeros_like(vel),
        vel
    )
    return pos, vel


def render_dot(
    pos: torch.Tensor,
    grid_size: int,
    radius: float,
    intensity: float,
    background: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Render a Gaussian-like dot on a 2D grid.
    pos: (B, 2) or (B, T, 2) - x, y coordinates
    Returns: (B, 1, H, W) or (B, T, 1, H, W)
    """
    squeeze_time = pos.dim() == 2
    if squeeze_time:
        pos = pos.unsqueeze(1)  # (B, 1, 2)

    B, T, _ = pos.shape
    y = torch.arange(grid_size, device=device, dtype=pos.dtype)
    x = torch.arange(grid_size, device=device, dtype=pos.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (H, W)

    # (B, T, 1, 1) vs (1, 1, H, W) for broadcasting to (B, T, H, W)
    px = pos[..., 0:1].unsqueeze(-1)  # (B, T, 1, 1)
    py = pos[..., 1:2].unsqueeze(-1)  # (B, T, 1, 1)

    xx_b = xx.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    yy_b = yy.unsqueeze(0).unsqueeze(0)
    dist_sq = (xx_b - px) ** 2 + (yy_b - py) ** 2
    img = intensity * torch.exp(-dist_sq / (2 * radius ** 2))
    img = img + background
    img = img.clamp(0, 1)

    if squeeze_time:
        # Keep (B, 1, H, W) - the 1 is channel dim for grayscale
        pass  # img already (B, 1, H, W) from (B, T, H, W) with T=1
    return img


def apply_occlusion(
    obs: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Apply occlusion mask to observations. obs: (B,T,C,H,W), mask: (B,T,1,H,W)."""
    return obs * (1 - mask) + fill_value * mask


def generate_occlusion_masks(
    batch_size: int,
    seq_len: int,
    grid_size: int,
    config: MovingDotConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Generate random rectangular occlusion masks.
    Returns: (B, T, 1, H, W) binary mask (1 = occluded)
    """
    masks = torch.zeros(batch_size, seq_len, 1, grid_size, grid_size, device=device)
    for b in range(batch_size):
        t = 0
        while t < seq_len:
            if rng.random() < config.occlusion_prob:
                n_frames = rng.integers(config.occlusion_min_frames, config.occlusion_max_frames + 1)
                n_frames = min(n_frames, seq_len - t)
                h = rng.integers(config.occlusion_min_size, config.occlusion_max_size + 1)
                w = rng.integers(config.occlusion_min_size, config.occlusion_max_size + 1)
                y0 = rng.integers(0, max(1, grid_size - h + 1))
                x0 = rng.integers(0, max(1, grid_size - w + 1))
                for i in range(n_frames):
                    if t + i < seq_len:
                        masks[b, t + i, 0, y0 : y0 + h, x0 : x0 + w] = 1.0
                t += n_frames
            else:
                t += 1
    return masks


def generate_moving_dot_sequences(
    batch_size: int,
    config: MovingDotConfig,
    device: torch.device,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, torch.Tensor]:
    """
    Generate a batch of moving-dot sequences with control.

    Returns dict with:
        observations: (B, T, 1, H, W) - possibly occluded
        actions: (B, T, 2) - (ax, ay)
        states: (B, T, 4) - (x, y, vx, vy) ground truth
        occlusion_masks: (B, T, 1, H, W) - 1 where occluded (if use_occlusion)
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    grid_size = config.grid_size
    seq_len = config.seq_len
    boundary_fn = _reflect_boundary if config.boundary == "reflect" else _clip_boundary

    # Initialize state: (B, 4) = (x, y, vx, vy)
    x = rng.uniform(2, grid_size - 3, size=(batch_size,)).astype(np.float32)
    y = rng.uniform(2, grid_size - 3, size=(batch_size,)).astype(np.float32)
    vx = rng.uniform(-config.max_velocity, config.max_velocity, size=(batch_size,)).astype(np.float32)
    vy = rng.uniform(-config.max_velocity, config.max_velocity, size=(batch_size,)).astype(np.float32)

    pos = torch.tensor(np.stack([x, y], axis=1), device=device, dtype=torch.float32)
    vel = torch.tensor(np.stack([vx, vy], axis=1), device=device, dtype=torch.float32)

    # Sample random actions for the whole sequence
    actions = torch.tensor(
        rng.uniform(-1, 1, size=(batch_size, seq_len, 2)).astype(np.float32),
        device=device,
    ) * config.action_scale

    obs_list = []
    state_list = []

    for t in range(seq_len):
        # Render frame (no occlusion yet)
        img = render_dot(
            pos, grid_size, config.dot_radius, config.dot_intensity, config.background, device
        )
        obs_list.append(img)
        state_list.append(torch.cat([pos, vel], dim=-1))

        if t < seq_len - 1:
            a = actions[:, t, :]  # (B, 2)
            vel = vel + a
            vel = vel.clamp(-config.max_velocity, config.max_velocity)
            pos = pos + vel
            pos, vel = boundary_fn(pos, vel, grid_size)

    observations = torch.stack(obs_list, dim=1)  # (B, T, 1, H, W)
    states = torch.stack(state_list, dim=1)  # (B, T, 4)

    # Apply occlusion if enabled
    occlusion_masks = None
    if config.use_occlusion:
        occlusion_masks = generate_occlusion_masks(
            batch_size, seq_len, grid_size, config, device, rng
        )
        observations = apply_occlusion(observations, occlusion_masks, config.background)

    return {
        "observations": observations,
        "actions": actions,
        "states": states,
        "occlusion_masks": occlusion_masks,
    }


class MovingDotDataset(torch.utils.data.Dataset):
    """Dataset that generates moving-dot sequences on the fly."""

    def __init__(self, num_samples: int, config: MovingDotConfig, seed: Optional[int] = None):
        self.num_samples = num_samples
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Use idx to seed for reproducibility within epoch
        rng = np.random.default_rng(self.seed + idx if self.seed is not None else None)
        data = generate_moving_dot_sequences(1, self.config, torch.device("cpu"), rng)
        return {k: v.squeeze(0) for k, v in data.items()}


def collate_moving_dot(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate batch of sequences."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        items = [b[k] for b in batch if b[k] is not None]
        if items:
            out[k] = torch.stack(items, dim=0)
        else:
            out[k] = None
    return out
