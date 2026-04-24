"""Training loop for PPO on MuJoCo Ant-v5.

Supports observation/reward normalization, linear learning-rate annealing,
optional domain randomization, and checkpoint save/resume.

Usage:
    python v1_state_ppo/train.py
    python v1_state_ppo/train.py --resume checkpoints/ant_ppo_final.pt
    python v1_state_ppo/train.py --steps 5000000 --device cuda --checkpoint-dir checkpoints
"""

import os

# Headless / cluster default; override with MUJOCO_GL=... before launch if needed.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import time
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import torch

from shared.domain_random import DomainRandomizer
from shared.obs_normalizer import ObsNormalizer
from shared.reward_normalizer import RewardNormalizer
from v1_state_ppo.ppo_agent import PPOAgent

INITIAL_LR = 3e-4
BUFFER_SIZE = 2048
LOG_INTERVAL = 5


def _build_checkpoint(
    agent: PPOAgent,
    obs_normalizer: ObsNormalizer,
    reward_normalizer: RewardNormalizer,
    global_step: int,
    update: int,
    episode_count: int,
    mean_reward: float = 0.0,
) -> dict[str, Any]:
    return {
        "network": agent.network.state_dict(),
        "model_state_dict": agent.network.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "obs_normalizer": {
            "mean": obs_normalizer.mean,
            "var": obs_normalizer.var,
            "count": obs_normalizer.count,
        },
        "reward_normalizer": {
            "mean": reward_normalizer.mean,
            "var": reward_normalizer.var,
            "count": reward_normalizer.count,
        },
        "global_step": global_step,
        "step": global_step,
        "update": update,
        "episode_count": episode_count,
        "reward": mean_reward,
    }


def save_checkpoint(
    agent: PPOAgent,
    obs_normalizer: ObsNormalizer,
    reward_normalizer: RewardNormalizer,
    path: str,
    step: int,
    update: int,
    episode_count: int,
    reward: float,
    logger: logging.Logger,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        _build_checkpoint(
            agent,
            obs_normalizer,
            reward_normalizer,
            step,
            update,
            episode_count,
            mean_reward=reward,
        ),
        path,
    )
    logger.info(
        "[Checkpoint] Saved at step %s, reward %.1f -> %s",
        f"{step:,}",
        reward,
        path,
    )


def load_checkpoint(
    path: str,
    agent: PPOAgent,
    obs_normalizer: ObsNormalizer,
    reward_normalizer: RewardNormalizer,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[int, int, int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    agent.network.load_state_dict(checkpoint["network"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])

    obs_normalizer.mean = checkpoint["obs_normalizer"]["mean"]
    obs_normalizer.var = checkpoint["obs_normalizer"]["var"]
    obs_normalizer.count = checkpoint["obs_normalizer"]["count"]

    reward_normalizer.mean = checkpoint["reward_normalizer"]["mean"]
    reward_normalizer.var = checkpoint["reward_normalizer"]["var"]
    reward_normalizer.count = checkpoint["reward_normalizer"]["count"]

    start_update = int(checkpoint["update"])
    start_step = int(checkpoint["global_step"])
    start_episodes = int(checkpoint["episode_count"])
    reward = float(checkpoint.get("reward", 0.0))
    logger.info(
        "[Checkpoint] Resumed from step %s, last logged reward %.1f",
        f"{start_step:,}",
        reward,
    )
    return start_update, start_step, start_episodes, reward


def _setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger(__name__)
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
        description="Train a PPO agent on MuJoCo Ant-v5 (state observations)."
    )
    parser.add_argument(
        "--steps", type=int, default=5_000_000, help="Total environment steps"
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
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100_000,
        help="Save checkpoint every N environment steps",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for training log files",
    )
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="Enable domain randomization",
    )
    args = parser.parse_args()

    logger = _setup_logger(args.log_dir)

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

    total_timesteps = args.steps
    save_dir = args.checkpoint_dir.rstrip("/") or "checkpoints"
    save_interval_steps = args.save_interval
    use_domain_randomization = args.domain_randomization
    resume_from = args.resume

    env = gym.make("Ant-v5")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    logger.info("=" * 60)
    logger.info("PPO Training — Ant-v5")
    logger.info("=" * 60)
    logger.info("Observation dim:      %s", obs_dim)
    logger.info("Action dim:           %s", act_dim)
    logger.info("Total timesteps:      %s", f"{total_timesteps:,}")
    logger.info("Buffer size:          %s", BUFFER_SIZE)
    logger.info(
        "Domain randomization: %s",
        "ON" if use_domain_randomization else "OFF",
    )
    logger.info("=" * 60)

    agent = PPOAgent(
        obs_dim,
        act_dim,
        lr=INITIAL_LR,
        buffer_size=BUFFER_SIZE,
        device=device,
    )
    randomizer = (
        DomainRandomizer() if use_domain_randomization else None
    )
    obs_normalizer = ObsNormalizer(obs_dim)
    reward_normalizer = RewardNormalizer()

    start_step = 0
    start_episodes = 0
    start_update = 0

    if resume_from is not None:
        logger.info("Loading checkpoint: %s", resume_from)
        start_update, start_step, start_episodes, _r = load_checkpoint(
            resume_from,
            agent,
            obs_normalizer,
            reward_normalizer,
            device,
            logger,
        )

    obs, _ = env.reset()
    if randomizer is not None:
        randomizer.randomize(env)

    episode_rewards = 0.0
    episode_count = start_episodes
    recent_rewards: list[float] = []

    os.makedirs(save_dir, exist_ok=True)

    num_updates = total_timesteps // BUFFER_SIZE
    global_step = start_step
    start_time = time.time()
    last_logged_reward = 0.0
    last_save_step = -1

    for update in range(start_update + 1, num_updates + 1):

        for _step in range(BUFFER_SIZE):
            obs_normalizer.update(obs)
            norm_obs = obs_normalizer.normalize(obs)
            action, log_prob, value = agent.select_action(norm_obs)

            next_obs, reward, terminated, truncated, _info = env.step(
                action
            )
            done = terminated or truncated

            reward_normalizer.update(reward)
            norm_reward = reward_normalizer.normalize(reward)

            agent.buffer.store(
                norm_obs, action, norm_reward, value, log_prob, float(done)
            )

            episode_rewards += reward
            global_step += 1

            if done:
                recent_rewards.append(episode_rewards)
                episode_count += 1
                episode_rewards = 0.0
                next_obs, _ = env.reset()

                if randomizer is not None:
                    randomizer.randomize(env)

            obs = next_obs

        with torch.no_grad():
            norm_obs = obs_normalizer.normalize(obs)
            obs_tensor = torch.as_tensor(
                norm_obs, dtype=torch.float32, device=device
            )
            last_value = agent.network.get_value(obs_tensor).item()

        agent.buffer.compute_advantages(
            last_value, agent.gamma, agent.lam
        )

        progress = update / num_updates
        new_lr = INITIAL_LR * (1.0 - progress)
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = new_lr

        agent.update()

        if save_interval_steps > 0 and global_step > 0:
            if global_step // save_interval_steps > max(
                0, last_save_step // save_interval_steps
            ):
                mean_r = (
                    float(np.mean(recent_rewards[-20:]))
                    if recent_rewards
                    else 0.0
                )
                path = os.path.join(
                    save_dir, f"ant_ppo_{global_step}.pt"
                )
                save_checkpoint(
                    agent,
                    obs_normalizer,
                    reward_normalizer,
                    path,
                    global_step,
                    update,
                    episode_count,
                    mean_r,
                    logger,
                )
                last_save_step = global_step

        if update % LOG_INTERVAL == 0 and len(recent_rewards) > 0:
            elapsed = time.time() - start_time
            mean_reward = float(np.mean(recent_rewards[-20:]))
            last_logged_reward = mean_reward
            fps = global_step / elapsed

            logger.info(
                "Update %4d/%d | Step %8s | Episodes %4d | "
                "FPS %6.0f | Mean reward %8.1f | LR %.2e | Time %6.0fs",
                update,
                num_updates,
                f"{global_step:,}",
                episode_count,
                fps,
                mean_reward,
                new_lr,
                elapsed,
            )

            recent_rewards = recent_rewards[-20:]

    env.close()

    final_mean = (
        float(np.mean(recent_rewards[-20:]))
        if recent_rewards
        else last_logged_reward
    )
    final_path = os.path.join(save_dir, "ant_ppo_final.pt")
    save_checkpoint(
        agent,
        obs_normalizer,
        reward_normalizer,
        final_path,
        global_step,
        num_updates,
        episode_count,
        final_mean,
        logger,
    )
    logger.info("Training complete. Final model saved to %s", final_path)


if __name__ == "__main__":
    train()
