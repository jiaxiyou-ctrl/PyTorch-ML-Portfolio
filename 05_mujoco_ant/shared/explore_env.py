"""Quick exploration of the MuJoCo Ant-v5 environment.

Prints environment metadata and runs a few episodes with random actions
to verify the setup before training.

Usage (from project root)::

    python shared/explore_env.py
"""

import gymnasium as gym


def explore() -> None:
    """Print environment info and run random-action episodes."""
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
    print("Environment verification complete. Ready to train!")


if __name__ == "__main__":
    explore()
