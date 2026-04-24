"""Smoke test for domain randomization.

Runs a few episodes with randomized physics and prints the sampled
parameters to verify that randomization is working correctly.

Usage:
    python -m pytest tests/test_domain_random.py
    # or: python tests/test_domain_random.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym

from shared.domain_random import DomainRandomizer


def test() -> None:
    """Run short episodes with randomized physics and print parameters."""
    env = gym.make("Ant-v5")
    randomizer = DomainRandomizer()

    print("=" * 60)
    print("Domain Randomization Test")
    print("=" * 60)

    for episode in range(5):
        _obs, _ = env.reset()
        randomizer.randomize(env)
        params = randomizer.get_current_params(env)

        total_reward = 0.0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        print(
            f"Episode {episode + 1}: "
            f"gravity={params['gravity_z']:>6.2f}, "
            f"friction={params['friction_0']:>5.2f}, "
            f"mass={params['mass_total']:>6.2f} | "
            f"reward={total_reward:>7.1f}, steps={steps:>4d}"
        )

    env.close()
    print("\nDomain randomization is working!")


if __name__ == "__main__":
    test()
