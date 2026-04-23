"""Multimodal Actor-Critic networks with FiLM fusion."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional


class FiLM(nn.Module):
    def __init__(self, feature_dim: int, context_dim: int) -> None:
        super().__init__()
        
        self.gamma_generator = nn.Linear(context_dim, feature_dim)
        self.beta_generator = nn.Linear(context_dim, feature_dim)
       
        nn.init.zeros_(self.gamma_generator.weight)   
        nn.init.ones_(self.gamma_generator.bias)      
        nn.init.zeros_(self.beta_generator.weight)   
        nn.init.zeros_(self.beta_generator.bias)     


    def forward(
        self, visual_features: torch.Tensor, language_features: torch.Tensor
        ) -> torch.Tensor:
            gamma = self.gamma_generator(language_features)
            beta = self.beta_generator(language_features)

            return gamma * visual_features + beta

class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 9, latent_dim: int = 256) -> None:
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
        
        conv_output = self.convs(image)
        flat = conv_output.reshape(conv_output.size(0), -1)
        return self.fc(flat)

class MultimodalActorCritic(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        act_dim: int = 8,
        visual_dim : int = 256,
        language_dim : int = 256,
    ) -> None:
        super().__init__()
        self.encoder = CNNEncoder(in_channels, visual_dim)
        self.film = FiLM(visual_dim, language_dim)
        self.actor_mean = nn.Sequential(
            nn.Linear(visual_dim, 256), # 256 -> 256
            nn.Tanh(),
            nn.Linear(256, act_dim), # 256 -> 8
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(visual_dim, 256), # 256 -> 256
            nn.Tanh(),
            nn.Linear(256, 1), # 256 -> 1
        )

    def get_fused_features(
        self, image: torch.Tensor, language_features: torch.Tensor
    ) -> torch.Tensor:
        visual_features = self.encoder(image)
        fused_features = self.film(visual_features, language_features)
        return fused_features

    def get_action_and_value(
        self,
        image: torch.Tensor,
        language_features: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self.get_fused_features(image, language_features)
        action_mean = self.actor_mean(fused)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        value = self.critic(fused).squeeze(-1)

        return action, log_prob, entropy, value

    def get_value(
        self,
        image: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.get_fused_features(image, language_features)
        return self.critic(fused).squeeze(-1)
        
