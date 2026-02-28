"""End-to-end test: PixelObsWrapper → PixelPPOAgent pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch

from pixel_wrapper import PixelObsWrapper
from pixel_ppo_agent import PixelPPOAgent


def test_full_pipeline():
    print("=" * 60)
    print("Pixel PPO Full Process Testing")
    print("=" * 60)

    # --- 1. create pixel environment ---
    raw_env = gym.make("Ant-v5", render_mode="rgb_array")
    env = PixelObsWrapper(raw_env, image_size=84, frame_stack=3)
    print(f"Environment created successfully")
    print(f"   observation space: {env.observation_space.shape}")  # (9, 84, 84)
    print(f"   action space: {env.action_space.shape}")       # (8,)

    # --- 2. create pixel agent ---
    obs_shape = env.observation_space.shape  # (9, 84, 84)
    act_dim = env.action_space.shape[0]      # 8

    agent = PixelPPOAgent(
        obs_shape=obs_shape,
        act_dim=act_dim,
        buffer_size=128,    # use a small buffer for quick testing (use 2048 for training)
        batch_size=32,
        update_epochs=2,    # run only 2 epochs for testing
    )
    print(f"Agent created successfully")

    # --- 3. collect one rollout of data ---
    obs, _ = env.reset()
    print(f"\n--- collect 128 steps of data ---")

    for step in range(128):
        action, log_prob, value = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.buffer.store(obs, action, reward, value, log_prob, float(done))

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    print(f"Data collection completed")

    # check the image format in the buffer
    print(f"   Buffer observation dtype: {agent.buffer.observations.dtype}")    # uint8
    print(f"   Buffer observation shape: {agent.buffer.observations.shape}")    # (128, 9, 84, 84)
    print(f"   Buffer observation range: [{agent.buffer.observations.min()}, "
          f"{agent.buffer.observations.max()}]")                         # [0, 255]

    # compute memory usage
    obs_memory_mb = agent.buffer.observations.nbytes / (1024 * 1024)
    print(f"   Buffer observation memory: {obs_memory_mb:.1f} MB")

    # --- 4. compute GAE ---
    with torch.no_grad():
        last_obs_tensor = torch.tensor(obs, dtype=torch.float32)
        last_value = agent.network.get_value(last_obs_tensor).item()

    agent.buffer.compute_advantages(last_value)
    print(f"GAE computation completed")

    # --- 5. run one PPO update ---
    agent.update()
    print(f"PPO update completed (no errors)")

    # --- 6. summary ---
    print(f"\n{'=' * 60}")
    print(f"Full pipeline test passed!")
    print(f"{'=' * 60}")
    print(f"\nIf buffer_size=2048, observation memory is approximately: "
          f"{2048 * 9 * 84 * 84 / (1024**2):.0f} MB (uint8)")
    print(f"Under the same conditions, float32 needs: "
          f"{2048 * 9 * 84 * 84 * 4 / (1024**2):.0f} MB")
    print(f"Saved {4}x memory!")

    env.close()


if __name__ == "__main__":
    test_full_pipeline()
