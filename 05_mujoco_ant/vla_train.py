"""VLA-PPO training script for multi-task Ant-v5 control."""

import os
import time

import gymnasium as gym
import numpy as np
import torch

from vla_ppo_agent import VLAPPOAgent
from vla_wrapper import PixelObsWrapper
from language_reward_wrapper import LanguageRewardWrapper

CONFIG = {
    "image_size": 84,
    "frame_stack": 3,
    "total_timesteps": 5_000_000,
    "buffer_size": 2048,
    "batch_size": 64,
    "update_epochs": 4,
    "lr_encoder": 5e-4,
    "lr_heads": 1e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_range": 0.1,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_augmentation": True,
    "normalize_reward": True,
    "log_interval": 1,
    "save_interval": 50,
    "save_dir": "checkpoints_vla_v4",
}


def make_vla_env(image_size: int = 84, frame_stack: int = 3) -> gym.Env:
    """Create an Ant-v5 environment with pixel obs and language rewards."""
    raw_env = gym.make("Ant-v5", render_mode="rgb_array")
    pixel_env = PixelObsWrapper(raw_env, image_size=image_size, frame_stack=frame_stack)
    vla_env = LanguageRewardWrapper(pixel_env)
    return vla_env


def train():
    """Main training loop: rollout collection → advantage computation → PPO update."""
    cfg = CONFIG

    print("=" * 60)
    print("VLA-PPO Training for Ant-v5")
    print("=" * 60)
    print(f"  Image:        {cfg['image_size']}×{cfg['image_size']}")
    print(f"  Frame stack:  {cfg['frame_stack']}")
   
    print(
        "  Tasks:        walk_forward, walk_backward, turn_left, turn_right, stand_still"
    )
    print(f"  Total steps:  {cfg['total_timesteps']:,}")
    print("=" * 60)

    env = make_vla_env(cfg["image_size"], cfg["frame_stack"])
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    agent = VLAPPOAgent(
        obs_shape=obs_shape,
        act_dim=act_dim,
        lr_encoder=cfg["lr_encoder"],
        lr_heads=cfg["lr_heads"],
        gamma=cfg["gamma"],
        lam=cfg["lam"],
        clip_range=cfg["clip_range"],
        entropy_coef=cfg["entropy_coef"],
        value_coef=cfg["value_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        update_epochs=cfg["update_epochs"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        use_augmentation=cfg["use_augmentation"],
        normalize_reward=cfg["normalize_reward"],
    )

    global_step = 0
    iteration = 0
    best_reward = -float("inf")

    episode_reward = 0.0
    episode_length = 0
    episode_rewards = []

    task_rewards: dict = {name: [] for name in agent.task_names}

    obs, info = env.reset()
    current_task = info["task_name"]

    train_start_time = time.time()
   
    print("   ->  Training started!")

    while global_step < cfg["total_timesteps"]:
        iteration += 1
        iter_start_time = time.time()

        # ── Rollout collection ───────────────────────────────────
        for _step in range(cfg["buffer_size"]):
            action, log_prob, value = agent.select_action(obs, current_task)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            task_idx = agent.get_task_index(current_task)

            normalized_reward = agent.normalize_reward(reward, current_task)

            agent.buffer.store(
                obs,
                action,
                normalized_reward,
                value,
                log_prob,
                float(done),
                task_idx,
            )

            episode_reward += reward
            episode_length += 1
            global_step += 1

            if done:
                episode_rewards.append(episode_reward)
                task_rewards[current_task].append(episode_reward)
                episode_reward = 0.0
                episode_length = 0
                obs, info = env.reset()
                current_task = info["task_name"]
            else:
                obs = next_obs

        # ── Advantage computation ─────────────────────────────────
        with torch.no_grad():
            last_obs_tensor = torch.tensor(
                obs, dtype=torch.float32
            ).unsqueeze(0)
            task_idx_tensor = torch.tensor(
                [agent.get_task_index(current_task)]
            )
            language_features = agent.get_language_features(task_idx_tensor)
            last_value = agent.network.get_value(
                last_obs_tensor, language_features
            ).item()

        agent.buffer.compute_advantages(
            last_value, gamma=cfg["gamma"], lam=cfg["lam"]
        )
        agent.update()

        iter_time = time.time() - iter_start_time
        total_time = time.time() - train_start_time

        if iteration % cfg["log_interval"] == 0 and len(episode_rewards) > 0:
            recent = episode_rewards[-10:]
            avg_reward = np.mean(recent)
            steps_per_sec = cfg["buffer_size"] / iter_time

            task_summary = ""
            for task_name in agent.task_names:
                task_list = task_rewards[task_name]
                if len(task_list) > 0:
                    task_summary += (
                        f"{task_name}: {np.mean(task_list[-5:]):.1f} | "
                    )

            print(
                f"Iter {iteration:4d} | "
                f"Step {global_step:>8,} | "
                f"Reward {avg_reward:7.1f} | "
                f"Episodes {len(episode_rewards):4d} | "
                f"Speed: {steps_per_sec:.0f} sps | "
                f"Time {total_time / 60:.1f} m | "
            )
            if task_summary:
                print(f"  Per-Task: {task_summary}")

            if avg_reward > best_reward:
                best_reward = avg_reward

        if iteration % cfg["save_interval"] == 0:
            os.makedirs(cfg["save_dir"], exist_ok=True)
            path = os.path.join(
                cfg["save_dir"], f"vla_ppo_iter_{iteration}.pt"
            )
            torch.save(
                {
                    "iteration": iteration,
                    "global_step": global_step,
                    "best_reward": best_reward,
                    "network_state_dict": agent.network.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "language_projection_state_dict": agent.language_encoder.projection.state_dict(),
                },
                path,
            )
            print(f"   ->  Checkpoint saved: {path}")

    total_time = time.time() - train_start_time
    
    print("Training complete!")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Total time: {total_time / 60:.1f} min")

    env.close()


if __name__ == "__main__":
    train()
