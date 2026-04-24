"""Running observation normalizer using Welford's online algorithm.

Maintains a running estimate of the observation mean and variance so that
every observation fed to the policy network is zero-centred and unit-scaled.
This is critical for stable training when different sensor channels span
vastly different numerical ranges (e.g. joint angles vs. contact forces).
"""

import numpy as np


class ObsNormalizer:
    """Incrementally normalizes observations to zero mean / unit variance.

    Uses Welford's parallel algorithm to update statistics from single
    observations or batches.

    Args:
        obs_dim: Dimensionality of the observation vector.
        clip:    Symmetric clip range applied after normalization.
    """

    def __init__(self, obs_dim: int, clip: float = 5.0) -> None:
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self.count: float = 1e-4
        self.clip = clip

    def update(self, obs: np.ndarray) -> None:
        """Update running statistics with a new observation (or batch).

        Args:
            obs: Observation array of shape ``(obs_dim,)`` or
                 ``(batch, obs_dim)``.
        """
        batch_mean = np.mean(obs, axis=0) if obs.ndim > 1 else obs
        batch_var = np.var(obs, axis=0) if obs.ndim > 1 else np.zeros_like(obs)
        batch_count = obs.shape[0] if obs.ndim > 1 else 1

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Return the normalized (and clipped) observation.

        Args:
            obs: Raw observation array of shape ``(obs_dim,)``.

        Returns:
            Normalized float32 array clipped to ``[-clip, clip]``.
        """
        normalized = (obs - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)
