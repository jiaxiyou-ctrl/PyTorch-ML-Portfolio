"""Multimodal actor-critic with FiLM-based vision-language fusion.

CNNEncoder extracts visual features, FiLMLayer modulates them with a
language embedding, and MultimodalActorCritic provides policy + value heads.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional
import numpy as np

class FiMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM).

    Produces per-feature gamma (scale) and beta (shift) from a language
    embedding, then applies:  out = gamma * visual_features + beta.
    """

    def __init__(self, feature_dim: int, language_dim: int) -> None:
        super().__init__()
        self.gamma_generator = nn.Linear(language_dim, feature_dim)
        self.beta_generator = nn.Linear(language_dim, feature_dim)
        
        # Identity init: gamma=1, beta=0 at start so FiLM is a no-op
        # until the language signal learns to modulate features.
        nn.init.ones_(self.gamma_generator.bias)
        nn.init.zeros_(self.gamma_generator.weight)
        nn.init.zeros_(self.beta_generator.bias)
        nn.init.ones_(self.beta_generator.weight)

    def forward(
        self, visual_features: torch.Tensor, language_features: torch.Tensor
    ) -> torch.Tensor:
        gamma = self.gamma_generator(language_features)
        beta = self.beta_generator(language_features)

        return gamma * visual_features + beta

class CNNEncoder(nn.Module):
    """Nature-DQN-style conv stack: 3 conv layers -> flatten -> FC + LayerNorm."""

    def __init__(self, in_channels: int = 9, feature_dim: int = 256) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        conv_out = self.conv_layers(images)
        flat = conv_out.reshape(conv_out.size(0), -1)
        return self.fc(flat)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        conv_out = self.conv_layers(images)
        flat = conv_out.reshape(conv_out.size(0), -1)
        return self.fc(flat)

class MultimodalActorCritic(nn.Module):
    """Full policy network: CNN encoder -> FiLM fusion -> actor & critic heads.

    The actor outputs a diagonal Gaussian; the critic outputs a scalar V(s).
    """

    def __init__(
        self,
        in_channels: int = 9,
        act_dim: int = 8,
        visual_dim: int = 256,
        language_dim: int = 256,
    ) -> None:
        super().__init__()

        self.encoder = CNNEncoder(in_channels, visual_dim)
        self.film = FiMLayer(visual_dim, language_dim)

        self.actor_mean = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim),
        )
        # Learnable exploration parameter (state-independent log std)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def get_fused_features(
        self, images: torch.Tensor, language_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode images and modulate with language via FiLM."""
        visual_features = self.encoder(images)
        fused =  self.film(visual_features, language_features)
        return fused

    def get_action_and_value(
        self,
        images: torch.Tensor,
        language_features: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) an action and return (action, log_prob, entropy, value)."""
        fused = self.get_fused_features(images, language_features)

        action_mean = self.actor_mean(fused)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        if actions is None:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(fused).squeeze(-1)
        return actions, log_probs, entropy, value

    def get_value(
        self, images: torch.Tensor, language_features: torch.Tensor
    ) -> torch.Tensor:
        """Return V(s) without computing the policy (used for bootstrapping)."""
        fused = self.get_fused_features(images, language_features)
        return self.critic(fused).squeeze(-1)











    


