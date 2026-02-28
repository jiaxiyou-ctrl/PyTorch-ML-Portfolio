"""Quick look at Ant-v5: print env info, run random episodes."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym


def explore() -> None:
    env = gym.make("Ant-v5", render_mode="human")

    print("=" * 60)
    print("Ant-v5 Environment Info")
    print("=" * 60)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Action range:      [{env.action_space.low[0]}, {env.action_space.high[0]}]")

    num_episodes = 3

    for episode in range(num_episodes):
        obs, _ = env.reset()
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
            f"[Random] Episode {episode + 1} — "
            f"Reward: {total_reward:.1f}, Steps: {steps}"
        )

    env.close()


if __name__ == "__main__":
    explore()
