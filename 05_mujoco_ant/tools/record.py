"""Record a trained Ant-v5 agent as MP4 video."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import gymnasium as gym
import numpy as np
import torch

from obs_normalizer import ObsNormalizer
from ppo_agent import PPOAgent


def record(
    checkpoint_path: str = "checkpoints/ant_ppo_final.pt",
    num_episodes: int = 3,
    max_steps_per_episode: int = 1000,
    video_folder: str = "videos",
) -> None:
    env = gym.make("Ant-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="ant_ppo",
        episode_trigger=lambda _ep: True,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim, act_dim)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if isinstance(checkpoint, dict) and "network" in checkpoint:
        agent.network.load_state_dict(checkpoint["network"])
        obs_normalizer = ObsNormalizer(obs_dim)
        obs_normalizer.mean = checkpoint["obs_normalizer"]["mean"]
        obs_normalizer.var = checkpoint["obs_normalizer"]["var"]
        obs_normalizer.count = checkpoint["obs_normalizer"]["count"]
        print(f"  Full checkpoint loaded (step {checkpoint['global_step']:,})")
    else:
        agent.network.load_state_dict(checkpoint)
        obs_normalizer = ObsNormalizer(obs_dim)
        print("  Weights only (no normalizer stats)")

    agent.network.eval()

    os.makedirs(video_folder, exist_ok=True)

    print(f"\nRecording {num_episodes} episodes...")
    print("-" * 50)

    all_rewards: list[float] = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        steps = 0

        for _step in range(max_steps_per_episode):
            norm_obs = obs_normalizer.normalize(obs)
            action, _, _ = agent.select_action(norm_obs)

            obs, reward, terminated, truncated, _info = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        all_rewards.append(episode_reward)
        print(f"  Episode {ep + 1}: reward = {episode_reward:>8.1f}, steps = {steps}")

    env.close()

    print("-" * 50)
    print(f"Mean reward: {np.mean(all_rewards):.1f}")
    print(f"Best reward: {np.max(all_rewards):.1f}")
    print(f"\nVideos saved in '{video_folder}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record a trained Ant-v5 agent as MP4 video."
    )
    parser.add_argument("--checkpoint", default="checkpoints/ant_ppo_final.pt")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    record(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
    )
