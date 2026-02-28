"""Smoke test for domain randomization."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym

from domain_random import DomainRandomizer


def test() -> None:
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


if __name__ == "__main__":
    test()
