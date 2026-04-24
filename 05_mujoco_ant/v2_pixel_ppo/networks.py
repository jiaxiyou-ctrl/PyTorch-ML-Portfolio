"""CNN encoder and actor–critic heads for pixel-based PPO (Ant 84×84)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Nature-style CNN for stacked RGB frames (CHW)."""

    def __init__(self, in_channels: int, latent_dim: int = 256) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        conv_out = self.convs(image)
        flat = conv_out.reshape(conv_out.size(0), -1)
        return self.fc(flat)


class PixelActorCritic(nn.Module):
    """Gaussian policy and value head on top of a CNN encoder."""

    def __init__(
        self,
        in_channels: int,
        act_dim: int,
        latent_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = CNNEncoder(in_channels, latent_dim)
        self.actor_mean = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return state values; training uses ``get_action_and_value``."""
        return self.get_value(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
        return self.critic(features).squeeze(-1)
