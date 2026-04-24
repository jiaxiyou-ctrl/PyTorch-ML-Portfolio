"""VLA-PPO agent for multi-task vision-language control."""

from typing import Tuple
import numpy as np
import torch
from torch import nn

from v2_pixel_ppo.augmentation import random_shift

from .language_encoder import LanguageEncoder
from .language_reward_wrapper import TASK_INSTRUCTIONS
from .multimodal_networks import MultimodalActorCritic
from .vla_ppo_buffer import VLAPPOBuffer  


class RunningMeanStd:
    """Online running mean and standard deviation tracker."""

    def __init__(self, shape: Tuple[int, ...]) -> None:
        """Initialize with given shape (used for future multi-dim support)."""
        self._shape = shape  
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: float) -> None:
        """Update running statistics with a new sample (Welford's algorithm)."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: float) -> float:
        """Normalize a value using current running statistics."""
        return x / (np.sqrt(self.var) + 1e-8)


class VLAPPOAgent:  
    """PPO agent with vision-language fusion for multi-task Ant control."""

    def __init__(
        self,
        obs_shape: Tuple[int, ...] = (9, 84, 84),
        act_dim: int = 8,
        lr_encoder: float = 1e-4,
        lr_heads: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.005,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 8,
        batch_size: int = 64,
        buffer_size: int = 2048,
        use_augmentation: bool = True,
        normalize_reward: bool = True,
        visual_dim: int = 256,
        language_dim: int = 256,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the VLA-PPO agent with all hyperparameters."""
        self.device = device or torch.device("cpu")
        in_channels = obs_shape[0]

        self.language_encoder = LanguageEncoder(output_dim=language_dim)
        self.language_encoder.to(self.device)

        self.task_names = list(TASK_INSTRUCTIONS.keys())
        self.task_texts = [
            TASK_INSTRUCTIONS[name]["text"] for name in self.task_names
        ]
        self.task_name_to_index = {
            name: i for i, name in enumerate(self.task_names)
        }
        self._precomputed_embeddings = self.language_encoder.precompute_all(
            self.task_texts
        )
        self.task_embeddings = torch.stack(
            [self._precomputed_embeddings[text] for text in self.task_texts]
        ).to(self.device)

        self.network = MultimodalActorCritic(
            in_channels,
            act_dim,
            visual_dim,
            language_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.encoder.parameters(),
                    "lr": lr_encoder,
                },
                {
                    "params": self.network.film.parameters(),
                    "lr": lr_heads,
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
                {
                    "params": self.language_encoder.projection.parameters(),
                    "lr": lr_heads,
                },
            ]
        )

        self.buffer = VLAPPOBuffer(
            buffer_size, obs_shape, act_dim, num_tasks=len(self.task_names)
        )

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

        self._use_reward_normalization = normalize_reward
        if normalize_reward:
            self.reward_normalizers = {
                name: RunningMeanStd(shape=(1,))
                for name in self.task_names
            }
        else:
            self.reward_normalizers = {}

    def get_task_index(self, task_name: str) -> int:
        """Return the integer index for a given task name."""
        return self.task_name_to_index[task_name]

    def get_language_features(self, task_indices: torch.Tensor) -> torch.Tensor:
        """Look up precomputed language embeddings by task index."""
        return self.task_embeddings[task_indices.to(self.device)]

    def normalize_reward(self, reward: float, task_name: str) -> float:
        """Normalize reward per-task using independent running statistics."""
        if task_name in self.reward_normalizers:
            rms = self.reward_normalizers[task_name]
            rms.update(reward)
            return rms.normalize(reward)
        return reward

    def select_action(
        self, obs: np.ndarray, task_name: str
    ) -> Tuple[np.ndarray, float, float]:
        """Select an action given an observation and task name."""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            task_index = self.get_task_index(task_name)
            language_features = self.task_embeddings[task_index].unsqueeze(0)

            action, log_prob, _, value = self.network.get_action_and_value(
                obs_tensor,
                language_features,
            )

            return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def update(self) -> None:
        """Run PPO update for multiple epochs over the collected rollout buffer."""
        for _epoch in range(self.update_epochs):

            for batch in self.buffer.get_balanced_batches(self.batch_size):
                obs = batch["observations"].to(self.device)
                if self.use_augmentation:
                    obs = random_shift(obs, pad=4)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)
                task_indices = batch["task_indices"].to(self.device)

                lang_features = self.get_language_features(task_indices)
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                _, new_log_probs, entropy, new_values = (
                    self.network.get_action_and_value(
                        obs,
                        lang_features,
                        actions,
                    )
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

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                all_params = list(self.network.parameters()) + list(
                    self.language_encoder.projection.parameters()
                )
                nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)

                self.optimizer.step()

        self.buffer.reset()
