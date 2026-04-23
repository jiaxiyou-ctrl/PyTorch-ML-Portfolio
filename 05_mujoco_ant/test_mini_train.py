"""Mini training test: run 2 PPO iterations to verify the full pipeline."""

import time
from vla_train import make_vla_env
from vla_ppo_agent import VLAPPOAgent

import numpy as np
import torch

def mini_train():
    """Run a minimal training loop (2 iterations) to verify everything works."""
    print("=" * 60)
    print("Mini Training Test (2 iterations)")
    print("=" * 60)

    # Setup
    env = make_vla_env(image_size=84, frame_stack=3)
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    agent = VLAPPOAgent(
        obs_shape=obs_shape,
        act_dim=act_dim,
        buffer_size=256,       # small buffer for quick testing
        batch_size=64,
        update_epochs=2,       
    )

    obs, info = env.reset()
    current_task = info["task_name"]

    # Iteration 1: Rollout
    print("\n[Iter 1] Collecting rollout...")
    start = time.time()

    for step in range(256):
        action, log_prob, value = agent.select_action(obs, current_task)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        task_idx = agent.get_task_index(current_task)
        normalized_reward = agent.normalize_reward(reward)

        agent.buffer.store(
            obs, action, normalized_reward, value, log_prob, float(done), task_idx
        )

        if done:
            obs, info = env.reset()
            current_task = info["task_name"]
        else:
            obs = next_obs

    rollout_time = time.time() - start
    print(f"  Rollout done: {rollout_time:.1f}s ({256/rollout_time:.0f} steps/sec)")

    # Iteration 1: Compute advantages
    with torch.no_grad():
        last_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        task_idx_t = torch.tensor([agent.get_task_index(current_task)])
        lang_feat = agent.get_language_features(task_idx_t)
        last_value = agent.network.get_value(last_obs, lang_feat).item()

    agent.buffer.compute_advantages(last_value)
    print("  Advantages computed")

    # Iteration 1: PPO Update
    start = time.time()
    agent.update()
    update_time = time.time() - start
    print(f"  PPO update done: {update_time:.1f}s")

    # Iteration 2: Repeat
    print("\n[Iter 2] Collecting rollout...")
    for step in range(256):
        action, log_prob, value = agent.select_action(obs, current_task)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        task_idx = agent.get_task_index(current_task)
        normalized_reward = agent.normalize_reward(reward)

        agent.buffer.store(
            obs, action, normalized_reward, value, log_prob, float(done), task_idx
        )

        if done:
            obs, info = env.reset()
            current_task = info["task_name"]
        else:
            obs = next_obs

    with torch.no_grad():
        last_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        task_idx_t = torch.tensor([agent.get_task_index(current_task)])
        lang_feat = agent.get_language_features(task_idx_t)
        last_value = agent.network.get_value(last_obs, lang_feat).item()

    agent.buffer.compute_advantages(last_value)
    agent.update()
    print("  Iteration 2 complete")

    # Summary
    print("\n" + "=" * 60)
    print("Mini training test PASSED!")
    print("=" * 60)
    print("The full pipeline works:")
    print("  Environment creation")
    print("  Observation rendering (pixels)")
    print("  Language encoding (CLIP)")
    print("  Action selection (Actor)")
    print("  Value estimation (Critic)")
    print("  Buffer storage & retrieval")
    print("  Advantage computation (GAE)")
    print("  PPO update (backprop)")
    print("  Data augmentation (random shift)")
    print("\nReady for full training:")
    print("  python vla_train.py")

    env.close()


if __name__ == "__main__":
    mini_train()
