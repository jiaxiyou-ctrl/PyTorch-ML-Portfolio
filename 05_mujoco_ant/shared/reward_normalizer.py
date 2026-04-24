"""Running reward normalizer using Welford's online algorithm.

Divides rewards by the running standard deviation to keep gradients on a
consistent scale throughout training, preventing early large-magnitude
rewards from dominating parameter updates.
"""

import numpy as np


class RewardNormalizer:
    """Incrementally normalizes scalar rewards by running standard deviation.

    Args:
        clip: Symmetric clip range applied after normalization.
    """

    def __init__(self, clip: float = 10.0) -> None:
        self.clip = clip
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = 1e-4

    def update(self, reward: float) -> None:
        """Update running statistics with a new reward sample.

        Args:
            reward: Scalar reward from the environment.
        """
        delta = reward - self.mean
        self.count += 1
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, reward: float) -> float:
        """Return the reward divided by the running standard deviation.

        Args:
            reward: Raw scalar reward.

        Returns:
            Normalized and clipped reward.
        """
        normalized = reward / np.sqrt(self.var + 1e-8)
        return float(np.clip(normalized, -self.clip, self.clip))
