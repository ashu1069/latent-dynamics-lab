"""
Minimal Recurrent State-Space Model (RSSM) for learning latent dynamics from pixels.

Architecture:
  obs -> Encoder -> embed
  (h, a) -> Prior -> z ~ N(prior_mean, prior_std)     [predict without seeing obs]
  (h, embed) -> Posterior -> z ~ N(post_mean, post_std) [infer from obs]
  h' = GRU(concat(z, a), h)
  (h, z) -> Decoder -> recon
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


def _conv_out_size(size: int, layers: int, kernel: int, stride: int, pad: int = 1) -> int:
    """Output size after `layers` conv layers."""
    for _ in range(layers):
        size = (size + 2 * pad - kernel) // stride + 1
    return size


@dataclass
class RSSMConfig:
    """Configuration for the RSSM model."""
    obs_channels: int = 1
    obs_height: int = 32
    obs_width: int = 32
    action_dim: int = 2
    embed_dim: int = 64
    deter_dim: int = 128
    stoch_dim: int = 32
    hidden_dim: int = 128
    cnn_channels: tuple = (32, 64, 64)
    cnn_kernel_size: int = 4
    cnn_stride: int = 2


class CNNEncoder(nn.Module):
    """Encode image observations to embedding. Handles (B,C,H,W) and (B,T,C,H,W)."""

    def __init__(self, config: RSSMConfig):
        super().__init__()
        self.config = config
        layers = []
        in_c = config.obs_channels
        for out_c in config.cnn_channels:
            layers += [
                nn.Conv2d(in_c, out_c, config.cnn_kernel_size, config.cnn_stride, 1),
                nn.ReLU(inplace=True),
            ]
            in_c = out_c
        self.conv = nn.Sequential(*layers)

        n_layers = len(config.cnn_channels)
        h = _conv_out_size(config.obs_height, n_layers, config.cnn_kernel_size, config.cnn_stride)
        w = _conv_out_size(config.obs_width, n_layers, config.cnn_kernel_size, config.cnn_stride)
        self.feat_size = in_c * h * w
        self.proj = nn.Linear(self.feat_size, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_sequence = x.dim() == 5
        if is_sequence:
            B, T = x.shape[0], x.shape[1]
            x = x.reshape(B * T, *x.shape[2:])
        out = self.conv(x)
        out = out.flatten(1)
        out = self.proj(out)
        if is_sequence:
            out = out.reshape(B, T, -1)
        return out


class CNNDecoder(nn.Module):
    """Decode latent (h,z) to reconstructed image. Handles (B,D) and (B,T,D)."""

    def __init__(self, config: RSSMConfig, input_dim: int):
        super().__init__()
        self.config = config
        n_layers = len(config.cnn_channels)
        self.feat_h = _conv_out_size(config.obs_height, n_layers, config.cnn_kernel_size, config.cnn_stride)
        self.feat_w = _conv_out_size(config.obs_width, n_layers, config.cnn_kernel_size, config.cnn_stride)
        self.feat_c = config.cnn_channels[-1]
        self.feat_size = self.feat_c * self.feat_h * self.feat_w
        self.proj = nn.Linear(input_dim, self.feat_size)

        layers = []
        in_c = self.feat_c
        for out_c in reversed(config.cnn_channels[:-1]):
            layers += [
                nn.ConvTranspose2d(in_c, out_c, config.cnn_kernel_size, config.cnn_stride, 1),
                nn.ReLU(inplace=True),
            ]
            in_c = out_c
        layers += [
            nn.ConvTranspose2d(in_c, config.obs_channels, config.cnn_kernel_size, config.cnn_stride, 1),
            nn.Sigmoid(),
        ]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squeeze = z.dim() == 2
        if squeeze:
            z = z.unsqueeze(1)
        B, T, D = z.shape
        z = z.reshape(B * T, D)
        out = self.proj(z)
        out = out.reshape(B * T, self.feat_c, self.feat_h, self.feat_w)
        out = self.deconv(out)
        out = out.reshape(B, T, self.config.obs_channels, self.config.obs_height, self.config.obs_width)
        if squeeze:
            out = out.squeeze(1)
        return out


class RSSM(nn.Module):
    """
    RSSM: learns latent dynamics from (obs, action) sequences.

    At each step t:
      1. Prior:   z_prior ~ p(z | h, a)     [predict without obs]
      2. Posterior: z ~ q(z | h, embed)     [infer from obs; training only]
      3. Decode:  recon = Decoder(h, z)
      4. Recur:   h' = GRU(concat(z, a), h)
    """

    def __init__(self, config: RSSMConfig):
        super().__init__()
        self.config = config

        self.encoder = CNNEncoder(config)
        self.decoder = CNNDecoder(config, config.deter_dim + config.stoch_dim)
        self.gru = nn.GRUCell(config.stoch_dim + config.action_dim, config.deter_dim)

        def make_stoch_net(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, config.hidden_dim),
                nn.ELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ELU(),
                nn.Linear(config.hidden_dim, config.stoch_dim * 2),
            )

        self.prior_net = make_stoch_net(config.deter_dim + config.action_dim)
        self.posterior_net = make_stoch_net(config.deter_dim + config.embed_dim)

    def _split_stoch(self, out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mean, log_std

    def _sample(self, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        return mean + log_std.exp() * torch.randn_like(mean)

    def _prior(self, h: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self._split_stoch(self.prior_net(torch.cat([h, a], dim=-1)))
        return self._sample(mean, log_std), mean, log_std

    def _posterior(self, h: torch.Tensor, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self._split_stoch(self.posterior_net(torch.cat([h, embed], dim=-1)))
        return self._sample(mean, log_std), mean, log_std

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        h_init: Optional[torch.Tensor] = None,
        use_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        obs: (B, T, C, H, W), actions: (B, T, action_dim)
        use_posterior=True for training (use obs), False for open-loop rollout.
        """
        B, T = obs.shape[0], obs.shape[1]
        device = obs.device
        embed = self.encoder(obs)

        h = h_init if h_init is not None else torch.zeros(B, self.config.deter_dim, device=device)

        z_list, prior_mean_list, prior_logstd_list, post_mean_list, post_logstd_list = [], [], [], [], []
        recons_list = []

        for t in range(T):
            a, embed_t = actions[:, t, :], embed[:, t, :]

            if use_posterior:
                z, post_mean, post_logstd = self._posterior(h, embed_t)
                _, prior_mean, prior_logstd = self._prior(h, a)
            else:
                z, prior_mean, prior_logstd = self._prior(h, a)
                post_mean, post_logstd = prior_mean, prior_logstd

            z_list.append(z)
            prior_mean_list.append(prior_mean)
            prior_logstd_list.append(prior_logstd)
            post_mean_list.append(post_mean)
            post_logstd_list.append(post_logstd)

            recon = self.decoder(torch.cat([h, z], dim=-1).unsqueeze(1)).squeeze(1)
            recons_list.append(recon)

            h = self.gru(torch.cat([z, a], dim=-1), h)

        return {
            "reconstructions": torch.stack(recons_list, dim=1),
            "z": torch.stack(z_list, dim=1),
            "prior_mean": torch.stack(prior_mean_list, dim=1),
            "prior_logstd": torch.stack(prior_logstd_list, dim=1),
            "post_mean": torch.stack(post_mean_list, dim=1),
            "post_logstd": torch.stack(post_logstd_list, dim=1),
            "h_final": h,
        }

    def open_loop_rollout(
        self,
        obs_context: torch.Tensor,
        actions_context: torch.Tensor,
        actions_future: torch.Tensor,
        K: int,
    ) -> dict[str, torch.Tensor]:
        """
        Condition on first K frames (posterior), then rollout open-loop (prior only).
        obs_context: (B, K, C, H, W), actions_context: (B, K, action_dim)
        actions_future: (B, T-K, action_dim)
        """
        B = obs_context.shape[0]
        device = obs_context.device
        embed = self.encoder(obs_context)
        h = torch.zeros(B, self.config.deter_dim, device=device)
        z_list, recons_list = [], []

        for t in range(K):
            a, embed_t = actions_context[:, t, :], embed[:, t, :]
            z, _, _ = self._posterior(h, embed_t)
            z_list.append(z)
            recons_list.append(self.decoder(torch.cat([h, z], dim=-1).unsqueeze(1)).squeeze(1))
            h = self.gru(torch.cat([z, a], dim=-1), h)

        for t in range(actions_future.shape[1]):
            a = actions_future[:, t, :]
            z, _, _ = self._prior(h, a)
            z_list.append(z)
            recons_list.append(self.decoder(torch.cat([h, z], dim=-1).unsqueeze(1)).squeeze(1))
            h = self.gru(torch.cat([z, a], dim=-1), h)

        return {
            "reconstructions": torch.stack(recons_list, dim=1),
            "z": torch.stack(z_list, dim=1),
        }


def kl_gaussian(mean1: torch.Tensor, logstd1: torch.Tensor, mean2: torch.Tensor, logstd2: torch.Tensor) -> torch.Tensor:
    """KL divergence between two Gaussians. Returns (B, T) or (B,)"""
    var1 = logstd1.exp().pow(2)
    var2 = logstd2.exp().pow(2)
    kl = 0.5 * (var1 / var2 + (mean2 - mean1).pow(2) / var2 - 1 + 2 * (logstd2 - logstd1))
    return kl.sum(dim=-1)
