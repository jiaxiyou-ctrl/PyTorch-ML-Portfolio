"""Actor-Critic network for PPO on MuJoCo Ant-v5.

Separate actor and critic trunks, each a two-layer MLP with Tanh
activations.  The actor outputs a diagonal Gaussian over joint torques;
the critic outputs a single scalar state-value estimate.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Shared Actor-Critic module with separate network trunks.

    Architecture::

        obs ─┬─► [Linear → Tanh → Linear → Tanh → Linear] ─► action mean
             │                                                 + learnable log_std
             └─► [Linear → Tanh → Linear → Tanh → Linear] ─► state value

    Args:
        obs_dim:    Dimensionality of the observation vector.
        act_dim:    Dimensionality of the action vector.
        hidden_dim: Width of hidden layers (default 256).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
        )

        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action (or evaluate a given one) and return value estimate.

        Args:
            obs:    Observation tensor of shape ``(*, obs_dim)``.
            action: If provided, evaluate this action instead of sampling.

        Returns:
            Tuple of ``(action, log_prob, entropy, value)``.
        """
        action_mean = self.actor_mean(obs)
        action_std = self.actor_log_std.exp()
        dist = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the critic's state-value estimate.

        Args:
            obs: Observation tensor of shape ``(*, obs_dim)``.

        Returns:
            Scalar value tensor of shape ``(*,)``.
        """
        return self.critic(obs).squeeze(-1)
