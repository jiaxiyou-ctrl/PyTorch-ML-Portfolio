"""PPO rollout buffer adapted for pixel observations.
Observations stored as uint8 to save 4x memory.
"""

from typing import Dict, Generator, Tuple

import numpy as np
import torch

class PixelPPOBuffer:
    """Fixed-size buffer for one rollout. Computes GAE before each update."""

    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], act_dim: int) -> None:
        self.observations = np.zeros(
            (buffer_size, *obs_shape), dtype=np.uint8
        )

        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.ptr = 0

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: float,
    ) -> None:
        """Append one transition."""
        # float32 [0,1] -> uint8 [0,255] to save memory
        self.observations[self.ptr] = (obs * 255).astype(np.uint8)

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1

    def compute_advantages(
        self, last_value: float, gamma=0.99, lam=0.95
    ) -> None:
        """Compute GAE-lambda advantages and returns."""
        last_gae = 0.0

        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )

            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(
        self, batch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield shuffled mini-batches as tensors."""
        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices)

        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            obs_unit8 = self.observations[batch_idx]
            obs_float = obs_unit8.astype(np.float32) / 255.0

            yield {
                "observations": torch.tensor(obs_float),
                "actions": torch.tensor(self.actions[batch_idx]),
                "log_probs": torch.tensor(self.log_probs[batch_idx]),
                "advantages": torch.tensor(self.advantages[batch_idx]),
                "returns": torch.tensor(self.returns[batch_idx]),
            }

    def reset(self) -> None:
        """Reset write pointer for next rollout."""
        self.ptr = 0


            