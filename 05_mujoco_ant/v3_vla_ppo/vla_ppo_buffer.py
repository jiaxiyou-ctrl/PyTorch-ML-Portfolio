"""PPO rollout buffer for VLA (Vision-Language-Action) training."""

from typing import Dict, Generator, Tuple
import numpy as np
import torch


class VLAPPOBuffer:
    """Fixed-size buffer for one rollout. Computes GAE before each update."""
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        act_dim: int,
        num_tasks: int = 5,
    ) -> None:
        """Initialize buffer arrays for all transition components."""
        self._num_tasks = num_tasks  

        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)

        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.task_indices = np.zeros(buffer_size, dtype=np.int64)
        self.buffer_size = buffer_size
        self.ptr = 0

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: float,
        task_index: int,
    ) -> None:
        """Append one transition to the buffer."""
        self.observations[self.ptr] = (obs * 255).astype(np.uint8)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.task_indices[self.ptr] = task_index
        self.ptr += 1

    def compute_advantages(
        self, last_value: float, gamma: float = 0.99, lam: float = 0.95
    ) -> None:
        """Compute GAE-Lambda advantages and discounted returns.

        Args:
            last_value: Bootstrap value estimate for the final state.
            gamma: Discount factor.
            lam: GAE lambda for bias-variance trade-off.
        """
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
        """Yield shuffled mini-batches of transitions as PyTorch tensors."""
        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices)

        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                "observations": torch.tensor(
                    self.observations[batch_idx], dtype=torch.float32
                )
                / 255.0,
                "actions": torch.tensor(self.actions[batch_idx]),
                "log_probs": torch.tensor(self.log_probs[batch_idx]),
                "advantages": torch.tensor(self.advantages[batch_idx]),
                "returns": torch.tensor(self.returns[batch_idx]),
                "task_indices": torch.tensor(self.task_indices[batch_idx]),
            }

    def get_balanced_batches(
        self, batch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield mini-batches with equal representation from each task."""
        unique_tasks = np.unique(self.task_indices[:self.buffer_size])
        num_tasks = len(unique_tasks)

    
        per_task = max(1, batch_size // num_tasks)

        task_to_indices = {}
        for task_id in unique_tasks:
            task_mask = self.task_indices[:self.buffer_size] == task_id
            task_to_indices[task_id] = np.where(task_mask)[0]

        min_task_samples = min(len(idx) for idx in task_to_indices.values())
        num_batches = max(1, min_task_samples // per_task)

        for b in range(num_batches):
            batch_indices = []
            for task_id in unique_tasks:
                start = b * per_task
                end = start + per_task
                indices = task_to_indices[task_id]
                selected = indices[start % len(indices):(start % len(indices)) + per_task]
                if len(selected) < per_task:
                    extra = indices[:per_task - len(selected)]
                    selected = np.concatenate([selected, extra])
                batch_indices.append(selected)
            batch_idx = np.concatenate(batch_indices)
            np.random.shuffle(batch_idx)
        yield {
            "observations": torch.tensor(
                self.observations[batch_idx], dtype=torch.float32
            ) / 255.0,
            "actions": torch.tensor(self.actions[batch_idx]),
            "log_probs": torch.tensor(self.log_probs[batch_idx]),
            "advantages": torch.tensor(self.advantages[batch_idx]),
            "returns": torch.tensor(self.returns[batch_idx]),
            "task_indices": torch.tensor(self.task_indices[batch_idx]),
        }

    
    def reset(self) -> None:
        """Reset the buffer pointer to prepare for a new rollout."""
        self.ptr = 0
