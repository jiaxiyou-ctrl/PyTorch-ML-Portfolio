"""Train PPO agent to control Ant from pixel observations."""

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

from v2_pixel_ppo.pixel_ppo_agent import PixelPPOAgent
from v2_pixel_ppo.pixel_wrapper import PixelObsWrapper

CONFIG: dict[str, Any] = {
    "image_size": 84,
    "frame_stack": 3,
    "buffer_size": 2048,
    "batch_size": 64,
    "update_epochs": 8,
    "lr_encoder": 1e-4,
    "lr_heads": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_range": 0.2,
    "entropy_coef": 0.005,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_augmentation": True,
    "normalize_reward": True,
    "log_interval": 1,
}


def make_pixel_env(image_size: int = 84, frame_stack: int = 3) -> gym.Env:
    """Create Ant-v5 wrapped with pixel observations and frame stacking."""
    raw_env = gym.make("Ant-v5", render_mode="rgb_array")
    pixel_env = PixelObsWrapper(
        raw_env, image_size=image_size, frame_stack=frame_stack
    )
    return pixel_env


def save_checkpoint(
    agent: PixelPPOAgent,
    save_dir: str,
    iteration: int,
    global_step: int,
    best_reward: float,
    path: str | None,
    logger: logging.Logger,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    out = path or os.path.join(
        save_dir, f"pixel_ppo_iter_{iteration}.pt"
    )
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
        },
        out,
    )
    logger.info(
        "[Checkpoint] Saved at step %s, reward %.1f -> %s",
        f"{global_step:,}",
        best_reward,
        out,
    )


def load_checkpoint(
    path: str,
    agent: PixelPPOAgent,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agent.network.load_state_dict(ckpt["network_state_dict"])
    agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
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
    logger = logging.getLogger("v2_pixel_ppo.pixel_train")
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
        description="Pixel-based PPO on Ant-v5."
    )
    parser.add_argument(
        "--steps", type=int, default=5_000_000, help="Total env steps"
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
        default="checkpoints_pixel",
        help="Checkpoint directory",
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
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Log directory"
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

    save_dir = args.checkpoint_dir.rstrip("/") or "checkpoints_pixel"
    save_interval_steps = args.save_interval
    total_timesteps = args.steps

    logger.info("=" * 60)
    logger.info("Pixel-based PPO Training for Ant-v5")
    logger.info("=" * 60)
    logger.info("  Image:        %s×%s", cfg["image_size"], cfg["image_size"])
    logger.info("  Frame stack:  %s", cfg["frame_stack"])
    logger.info("  Buffer size:  %s", cfg["buffer_size"])
    logger.info("  Augmentation: %s", cfg["use_augmentation"])
    logger.info("  Total steps:  %s", f"{total_timesteps:,}")
    logger.info(
        "  Iterations:   ~%s",
        total_timesteps // cfg["buffer_size"],
    )
    logger.info("=" * 60)

    env = make_pixel_env(cfg["image_size"], cfg["frame_stack"])
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    agent = PixelPPOAgent(
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

    total_params = sum(p.numel() for p in agent.network.parameters())

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

    obs, _ = env.reset()
    train_start_time = time.time()

    logger.info("   ->  Training started! (params: %s)", f"{total_params:,}")

    while global_step < total_timesteps:
        iteration += 1
        iter_start_time = time.time()

        for _step in range(cfg["buffer_size"]):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _info = env.step(
                action
            )
            done = terminated or truncated

            normalized_reward = agent.normalize_rew(reward)
            agent.buffer.store(
                obs, action, normalized_reward, value, log_prob, float(done)
            )
            episode_reward += reward
            episode_length += 1
            global_step += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                episode_length = 0
                obs, _ = env.reset()
            else:
                obs = next_obs

        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            )
            last_value = agent.network.get_value(last_obs_tensor).item()

        agent.buffer.compute_advantages(
            last_value, gamma=cfg["gamma"], lam=cfg["lam"]
        )

        agent.update()

        iter_time = time.time() - iter_start_time
        total_time = time.time() - train_start_time

        if iteration % cfg["log_interval"] == 0 and len(episode_rewards) > 0:
            recent = episode_rewards[-10:]
            avg_reward = float(np.mean(recent))
            min_reward = float(np.min(recent))
            max_reward = float(np.max(recent))
            steps_per_sec = cfg["buffer_size"] / iter_time
            logger.info(
                "Iter %4d | Step %8s | Reward %7.1f | min=%.0f, max=%.0f | "
                "Episodes %4d | Speed: %.0f steps/s | Time %.1f min",
                iteration,
                f"{global_step:,}",
                avg_reward,
                min_reward,
                max_reward,
                len(episode_rewards),
                steps_per_sec,
                total_time / 60,
            )
            if avg_reward > best_reward:
                best_reward = avg_reward

        if save_interval_steps > 0 and global_step > 0:
            if global_step // save_interval_steps > max(
                0, last_save_step // save_interval_steps
            ):
                path = os.path.join(
                    save_dir, f"pixel_ppo_{global_step}.pt"
                )
                save_checkpoint(
                    agent,
                    save_dir,
                    iteration,
                    global_step,
                    best_reward,
                    path,
                    logger,
                )
                last_save_step = global_step

    total_time = time.time() - train_start_time
    logger.info("Training complete!")
    logger.info("Best reward: %.1f", best_reward)
    logger.info("Total steps: %s", f"{global_step:,}")
    logger.info("Total episodes: %s", f"{len(episode_rewards):,}")
    logger.info("Total time: %.1f min", total_time / 60)
    logger.info("Total params: %s", f"{total_params:,}")

    final_path = os.path.join(save_dir, "pixel_ppo_final.pt")
    save_checkpoint(
        agent,
        save_dir,
        iteration,
        global_step,
        best_reward,
        final_path,
        logger,
    )
    env.close()


if __name__ == "__main__":
    train()
