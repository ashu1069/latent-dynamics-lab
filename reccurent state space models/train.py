import argparse
import os
from pathlib import Path

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import MovingDotConfig, generate_moving_dot_sequences, collate_moving_dot
from model import RSSM, RSSMConfig, kl_gaussian
from utils import set_seed, kl_balance


def train_step(
    model: RSSM,
    obs: torch.Tensor,
    actions: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    free_bits: float = 1.0,
    kl_weight: float = 0.1,
    kl_balance_weight: float = 0.8,
) -> dict[str, float]:
    """Single training step. Loss = recon + kl_weight * kl_balanced."""
    model.train()
    optimizer.zero_grad()

    out = model(obs, actions, use_posterior=True)
    recon_loss = F.mse_loss(out["reconstructions"], obs)

    kl = kl_gaussian(
        out["post_mean"], out["post_logstd"],
        out["prior_mean"], out["prior_logstd"],
    ).mean()
    kl_balanced_val = kl_balance(kl.unsqueeze(0), free_bits, kl_balance_weight).squeeze(0)

    loss = recon_loss + kl_weight * kl_balanced_val
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    optimizer.step()

    return {"loss": loss.item(), "recon_loss": recon_loss.item(), "kl_loss": kl.item()}


def visualize_reconstruction(
    obs: torch.Tensor,
    recon: torch.Tensor,
    save_path: Path,
    n_frames: int = 8,
) -> None:
    """Plot input vs reconstruction for a batch sample."""
    obs = obs.detach().cpu()
    recon = recon.detach().cpu()
    # Use first sample
    obs = obs[0]  # (T, C, H, W)
    recon = recon[0]  # (T, C, H, W)
    T = min(obs.shape[0], n_frames)
    fig, axes = plt.subplots(2, T, figsize=(2 * T, 4))
    if T == 1:
        axes = axes.reshape(2, 1)
    for t in range(T):
        axes[0, t].imshow(obs[t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"Input t={t}")
        axes[0, t].axis("off")
        axes[1, t].imshow(recon[t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_title(f"Recon t={t}")
        axes[1, t].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def visualize_rollout(
    obs_true: torch.Tensor,
    recon_rollout: torch.Tensor,
    save_path: Path,
    K: int = 8,
    n_show: int = 16,
) -> None:
    """Plot true trajectory vs open-loop rollout."""
    obs_true = obs_true.detach().cpu()[0]  # (T, C, H, W)
    recon_rollout = recon_rollout.detach().cpu()[0]  # (T, C, H, W)
    T = min(obs_true.shape[0], n_show)
    fig, axes = plt.subplots(2, T, figsize=(2 * T, 4))
    if T == 1:
        axes = axes.reshape(2, 1)
    for t in range(T):
        axes[0, t].imshow(obs_true[t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"True t={t}")
        axes[0, t].axis("off")
        axes[1, t].imshow(recon_rollout[t, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_title(f"Rollout t={t}")
        axes[1, t].axis("off")
    plt.suptitle(f"Open-loop rollout (conditioned on first K={K} frames)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_occlusion", action="store_true", default=True)
    parser.add_argument("--no_occlusion", action="store_false", dest="use_occlusion")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--viz_interval", type=int, default=10)
    parser.add_argument("--K", type=int, default=8, help="Frames to condition on for open-loop")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data config
    data_config = MovingDotConfig(
        grid_size=args.grid_size,
        seq_len=args.seq_len,
        use_occlusion=args.use_occlusion,
        seed=args.seed,
    )

    # Model config
    model_config = RSSMConfig(
        obs_channels=1,
        obs_height=args.grid_size,
        obs_width=args.grid_size,
        action_dim=2,
        embed_dim=64,
        deter_dim=128,
        stoch_dim=32,
        hidden_dim=128,
    )

    model = RSSM(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    hist = {"loss": [], "recon_loss": [], "kl_loss": []}
    rng = __import__("numpy").random.default_rng(args.seed)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for _ in range(50):  # 50 batches per epoch
            data = generate_moving_dot_sequences(args.batch_size, data_config, device, rng)
            obs = data["observations"]
            actions = data["actions"]

            metrics = train_step(model, obs, actions, optimizer)
            epoch_loss += metrics["loss"]
            epoch_recon += metrics["recon_loss"]
            epoch_kl += metrics["kl_loss"]
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        hist["loss"].append(avg_loss)
        hist["recon_loss"].append(avg_recon)
        hist["kl_loss"].append(avg_kl)

        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

        # Visualization
        if (epoch + 1) % args.viz_interval == 0:
            model.eval()
            with torch.no_grad():
                # Get a batch for visualization
                data = generate_moving_dot_sequences(4, data_config, device, rng)
                obs = data["observations"]
                actions = data["actions"]

                # Reconstruction
                out = model(obs, actions, use_posterior=True)
                visualize_reconstruction(
                    obs, out["reconstructions"],
                    out_dir / f"recon_epoch{epoch + 1}.png",
                )

                # Open-loop rollout
                K = min(args.K, obs.shape[1] - 1)
                obs_ctx = obs[:, :K]
                actions_ctx = actions[:, :K]
                actions_future = actions[:, K:]
                rollout = model.open_loop_rollout(obs_ctx, actions_ctx, actions_future, K)
                visualize_rollout(
                    obs, rollout["reconstructions"],
                    out_dir / f"rollout_epoch{epoch + 1}.png",
                    K=K,
                )

    # Save model
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"Model saved to {out_dir / 'model.pt'}")

    # Evaluation: open-loop prediction and trajectory comparison
    print("Running evaluation...")
    model.eval()
    with torch.no_grad():
        data = generate_moving_dot_sequences(16, data_config, device, rng)
        obs = data["observations"]
        actions = data["actions"]
        K = args.K
        rollout = model.open_loop_rollout(
            obs[:, :K], actions[:, :K], actions[:, K:], K
        )
        rollout_recon = rollout["reconstructions"]
        eval_mse = F.mse_loss(rollout_recon, obs).item()
        print(f"Evaluation open-loop MSE: {eval_mse:.4f}")
        visualize_rollout(
            obs, rollout_recon,
            out_dir / "eval_rollout.png",
            K=K,
            n_show=min(24, obs.shape[1]),
        )
        print(f"Evaluation rollout saved to {out_dir / 'eval_rollout.png'}")

    # Plot loss curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(hist["loss"], label="Total")
    ax.plot(hist["recon_loss"], label="Recon")
    ax.plot(hist["kl_loss"], label="KL")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_title("Training Loss")
    plt.savefig(out_dir / "loss_curves.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Loss curves saved to {out_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
