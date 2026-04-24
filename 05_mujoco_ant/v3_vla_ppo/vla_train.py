"""VLA-PPO training script for multi-task Ant-v5 control."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from v3_vla_ppo.language_reward_wrapper import LanguageRewardWrapper
from v3_vla_ppo.vla_ppo_agent import VLAPPOAgent
from v3_vla_ppo.vla_wrapper import PixelObsWrapper

CONFIG: dict[str, Any] = {
    "image_size": 84,
    "frame_stack": 3,
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
}


def make_vla_env(
    image_size: int = 84, frame_stack: int = 3
) -> gym.Env:
    """Create an Ant-v5 environment with pixel obs and language rewards."""
    raw_env = gym.make("Ant-v5", render_mode="rgb_array")
    pixel_env = PixelObsWrapper(
        raw_env, image_size=image_size, frame_stack=frame_stack
    )
    vla_env = LanguageRewardWrapper(pixel_env)
    return vla_env


def save_checkpoint(
    agent: VLAPPOAgent,
    path: str,
    iteration: int,
    global_step: int,
    best_reward: float,
    logger: logging.Logger,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "global_step": global_step,
            "step": global_step,
            "best_reward": best_reward,
            "reward": best_reward,
            "network_state_dict": agent.network.state_dict(),
            "model_state_dict": agent.network.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "language_projection_state_dict": (
                agent.language_encoder.projection.state_dict()
            ),
        },
        path,
    )
    logger.info(
        "[Checkpoint] Saved at step %s, reward %.1f -> %s",
        f"{global_step:,}",
        best_reward,
        path,
    )


def load_checkpoint(
    path: str, agent: VLAPPOAgent, device: torch.device, logger: logging.Logger
) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agent.network.load_state_dict(ckpt["network_state_dict"])
    agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "language_projection_state_dict" in ckpt:
        agent.language_encoder.projection.load_state_dict(
            ckpt["language_projection_state_dict"]
        )
    it = int(ckpt.get("iteration", 0))
    gs = int(ckpt.get("global_step", ckpt.get("step", 0)))
    br = float(ckpt.get("best_reward", ckpt.get("reward", 0.0)))
    logger.info(
        "[Checkpoint] Resumed from step %s, reward %.1f",
        f"{gs:,}",
        br,
    )
    return it, gs, br


def _setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger("v3_vla_ppo.vla_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def train() -> None:
    parser = argparse.ArgumentParser(
        description="VLA-PPO training on Ant-v5."
    )
    parser.add_argument(
        "--steps", type=int, default=10_000_000, help="Total env steps"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cuda, or cpu",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_vla_v4",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100_000,
        help="Save every N environment steps",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Log file directory"
    )
    args = parser.parse_args()

    logger = _setup_logger(args.log_dir)
    cfg = dict(CONFIG)

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("VRAM: %.1f GB", vram)

    save_dir = args.checkpoint_dir.rstrip("/") or "checkpoints_vla_v4"
    save_interval_steps = args.save_interval
    total_timesteps = args.steps

    logger.info("=" * 60)
    logger.info("VLA-PPO Training for Ant-v5")
    logger.info("=" * 60)
    logger.info("  Image:        %s×%s", cfg["image_size"], cfg["image_size"])
    logger.info("  Frame stack:  %s", cfg["frame_stack"])
    logger.info(
        "  Tasks:        walk_forward, walk_backward, turn_left, "
        "turn_right, stand_still"
    )
    logger.info("  Total steps:  %s", f"{total_timesteps:,}")
    logger.info("=" * 60)

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
        device=device,
    )

    global_step = 0
    iteration = 0
    best_reward = -float("inf")
    last_save_step = -1

    if args.resume:
        iteration, global_step, best_reward = load_checkpoint(
            args.resume, agent, device, logger
        )

    episode_reward = 0.0
    episode_length = 0
    episode_rewards: list[float] = []

    task_rewards: dict = {name: [] for name in agent.task_names}

    obs, info = env.reset()
    current_task = info["task_name"]

    train_start_time = time.time()
    logger.info("   ->  Training started!")

    while global_step < total_timesteps:
        iteration += 1
        iter_start_time = time.time()

        for _step in range(cfg["buffer_size"]):
            action, log_prob, value = agent.select_action(
                obs, current_task
            )

            next_obs, reward, terminated, truncated, info = env.step(
                action
            )
            done = terminated or truncated

            task_idx = agent.get_task_index(current_task)

            normalized_reward = agent.normalize_reward(
                reward, current_task
            )

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

        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            task_idx_tensor = torch.tensor(
                [agent.get_task_index(current_task)], device=device
            )
            language_features = agent.get_language_features(
                task_idx_tensor
            )
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
            avg_reward = float(np.mean(recent))
            steps_per_sec = cfg["buffer_size"] / iter_time
            task_summary = ""
            for task_name in agent.task_names:
                tlist = task_rewards[task_name]
                if len(tlist) > 0:
                    task_summary += f"{task_name}: {np.mean(tlist[-5:]):.1f} | "
            logger.info(
                "Iter %4d | Step %8s | Reward %7.1f | Episodes %4d | "
                "Speed: %.0f sps | Time %.1f m |",
                iteration,
                f"{global_step:,}",
                avg_reward,
                len(episode_rewards),
                steps_per_sec,
                total_time / 60,
            )
            if task_summary:
                logger.info("  Per-Task: %s", task_summary)
            if avg_reward > best_reward:
                best_reward = avg_reward

        if save_interval_steps > 0 and global_step > 0:
            if global_step // save_interval_steps > max(
                0, last_save_step // save_interval_steps
            ):
                path = os.path.join(
                    save_dir, f"vla_ppo_{global_step}.pt"
                )
                save_checkpoint(
                    agent, path, iteration, global_step, best_reward, logger
                )
                last_save_step = global_step

    total_time = time.time() - train_start_time
    logger.info("Training complete!")
    logger.info("Best reward: %.1f", best_reward)
    logger.info("Total time: %.1f min", total_time / 60)

    final_path = os.path.join(save_dir, "vla_ppo_final.pt")
    save_checkpoint(
        agent, final_path, iteration, global_step, best_reward, logger
    )
    env.close()


if __name__ == "__main__":
    train()
