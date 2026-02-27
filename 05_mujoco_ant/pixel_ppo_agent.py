"""PPO agent adapted for pixel observations."""

from typing import Tuple

import numpy as np
import torch
from torch import nn

from networks import PixelActorCritic
from pixel_ppo_buffer import PixelPPOBuffer

class PixelPPOAgent:

    def __init__(
        self,
        obs_shape: Tuple[int, ...] = (9, 84, 84),
        act_dim: int = 8,
        lr_encoder: float = 1e-4,     
        lr_heads: float = 3e-4,     
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        use_augmentation: bool = True,
    ) -> None:
        in_channels = obs_shape[0]
        self.network = PixelActorCritic(in_channels, act_dim)
        self.optimizer = torch.optim.Adam([
            {
                "params": self.network.encoder.parameters(),
                "lr": lr_encoder,
            },
            {
                "params": self.network.actor_mean.parameters(),
                "lr": lr_heads,
            },

            {
                "params": [self.network.actor_log_std],
                "lr": lr_heads,
            },
            {
                "params": self.network.critic.parameters(),
                "lr": lr_heads,
            },
        ])

        self.buffer = PixelPPOBuffer(buffer_size, obs_shape, act_dim)
        self.use_augmentation = use_augmentation
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def select_action(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Choose action given a pixel observation."""

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action, log_prob, _, value = self.network.get_action_and_value(
                obs_tensor
            )
            return action.squeeze(0).numpy(), log_prob.item(), value.item()

    def update(self) -> None:
        """Run multiple epochs of PPO-Clip updates on the current buffer."""

        for _epoch in range(self.update_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs = batch["observations"]
                if self.use_augmentation:
                    obs = random_shift(obs, pad=4)
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                _, new_log_probs, entropy, new_values = (
                    self.network.get_action_and_value(obs, actions)
                )

                ratio = (new_log_probs - old_log_probs).exp()

                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1 - self.clip_range, 1 + self.clip_range
                    )
                    * advantages
                )

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (new_values - returns).pow(2).mean()
                entropy_loss = -entropy.mean()
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

        self.buffer.reset()

       
        
    
    