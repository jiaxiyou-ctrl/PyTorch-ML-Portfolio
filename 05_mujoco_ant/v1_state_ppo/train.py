"""Training loop for PPO on MuJoCo Ant-v5.

Supports observation/reward normalization, linear learning-rate annealing,
optional domain randomization, and checkpoint save/resume.

Usage:
    python v1_state_ppo/train.py                                        # fresh start
    python v1_state_ppo/train.py --resume checkpoints/ant_ppo_final.pt  # resume
    python v1_state_ppo/train.py --timesteps 5000000                    # custom length
"""

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch

from shared.domain_random import DomainRandomizer
from shared.obs_normalizer import ObsNormalizer
from shared.reward_normalizer import RewardNormalizer
from v1_state_ppo.ppo_agent import PPOAgent

# ── Default hyperparameters ───────────────────────────────────────────
INITIAL_LR = 3e-4
BUFFER_SIZE = 2048
LOG_INTERVAL = 5
SAVE_INTERVAL = 50


def _build_checkpoint(
    agent: PPOAgent,
    obs_normalizer: ObsNormalizer,
    reward_normalizer: RewardNormalizer,
    global_step: int,
    update: int,
    episode_count: int,
) -> dict:
    """Assemble a checkpoint dictionary for ``torch.save``."""
    return {
        "network": agent.network.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
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
        "update": update,
        "episode_count": episode_count,
    }


def train(
    total_timesteps: int = 1_000_000,
    buffer_size: int = BUFFER_SIZE,
    log_interval: int = LOG_INTERVAL,
    save_interval: int = SAVE_INTERVAL,
    save_dir: str = "checkpoints",
    use_domain_randomization: bool = False,
    resume_from: str | None = None,
) -> None:
    """Run the PPO training loop on Ant-v5.

    Args:
        total_timesteps:           Total environment steps to train for.
        buffer_size:               Rollout length before each PPO update.
        log_interval:              Print metrics every N updates.
        save_interval:             Save a checkpoint every N updates.
        save_dir:                  Directory for checkpoint files.
        use_domain_randomization:  Randomize physics each episode.
        resume_from:               Path to a checkpoint to resume from.
    """
    env = gym.make("Ant-v5")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print("=" * 60)
    print("PPO Training — Ant-v5")
    print("=" * 60)
    print(f"Observation dim:      {obs_dim}")
    print(f"Action dim:           {act_dim}")
    print(f"Total timesteps:      {total_timesteps:,}")
    print(f"Buffer size:          {buffer_size}")
    print(f"Domain randomization: {'ON' if use_domain_randomization else 'OFF'}")
    print("=" * 60)

    agent = PPOAgent(obs_dim, act_dim, lr=INITIAL_LR, buffer_size=buffer_size)
    randomizer = DomainRandomizer() if use_domain_randomization else None
    obs_normalizer = ObsNormalizer(obs_dim)
    reward_normalizer = RewardNormalizer()

    start_step = 0
    start_episodes = 0
    start_update = 0

    if resume_from is not None:
        print(f"Loading checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, weights_only=False)

        agent.network.load_state_dict(checkpoint["network"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])

        obs_normalizer.mean = checkpoint["obs_normalizer"]["mean"]
        obs_normalizer.var = checkpoint["obs_normalizer"]["var"]
        obs_normalizer.count = checkpoint["obs_normalizer"]["count"]

        reward_normalizer.mean = checkpoint["reward_normalizer"]["mean"]
        reward_normalizer.var = checkpoint["reward_normalizer"]["var"]
        reward_normalizer.count = checkpoint["reward_normalizer"]["count"]

        start_update = checkpoint["update"]
        start_step = checkpoint["global_step"]
        start_episodes = checkpoint["episode_count"]

        print(f"Resumed from update {start_update}, step {start_step:,}")

    obs, _ = env.reset()
    if randomizer is not None:
        randomizer.randomize(env)

    episode_rewards = 0.0
    episode_count = start_episodes
    recent_rewards: list[float] = []

    os.makedirs(save_dir, exist_ok=True)

    num_updates = total_timesteps // buffer_size
    global_step = start_step
    start_time = time.time()

    for update in range(start_update + 1, num_updates + 1):

        # ── Collect rollout ───────────────────────────────────
        for _step in range(buffer_size):
            obs_normalizer.update(obs)
            norm_obs = obs_normalizer.normalize(obs)
            action, log_prob, value = agent.select_action(norm_obs)

            next_obs, reward, terminated, truncated, _info = env.step(action)
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

        # ── Bootstrap value for the last state ────────────────
        with torch.no_grad():
            norm_obs = obs_normalizer.normalize(obs)
            obs_tensor = torch.tensor(norm_obs, dtype=torch.float32)
            last_value = agent.network.get_value(obs_tensor).item()

        agent.buffer.compute_advantages(last_value, agent.gamma, agent.lam)

        # ── Linear LR annealing to zero ──────────────────────
        progress = update / num_updates
        new_lr = INITIAL_LR * (1.0 - progress)
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = new_lr

        agent.update()

        # ── Logging ───────────────────────────────────────────
        if update % log_interval == 0 and len(recent_rewards) > 0:
            elapsed = time.time() - start_time
            mean_reward = np.mean(recent_rewards[-20:])
            fps = global_step / elapsed

            print(
                f"Update {update:>4d}/{num_updates} | "
                f"Step {global_step:>8,} | "
                f"Episodes {episode_count:>4d} | "
                f"FPS {fps:>6.0f} | "
                f"Mean reward {mean_reward:>8.1f} | "
                f"LR {new_lr:.2e} | "
                f"Time {elapsed:>6.0f}s"
            )

            recent_rewards = recent_rewards[-20:]

        # ── Periodic checkpoint ───────────────────────────────
        if update % save_interval == 0:
            path = os.path.join(save_dir, f"ant_ppo_{global_step}.pt")
            torch.save(
                _build_checkpoint(
                    agent, obs_normalizer, reward_normalizer,
                    global_step, update, episode_count,
                ),
                path,
            )
            print(f"   ->  Model saved to {path}")

    env.close()

    # ── Final checkpoint ──────────────────────────────────────
    final_path = os.path.join(save_dir, "ant_ppo_final.pt")
    torch.save(
        _build_checkpoint(
            agent, obs_normalizer, reward_normalizer,
            global_step, num_updates, episode_count,
        ),
        final_path,
    )
    print(f"Training complete. Final model saved to {final_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on MuJoCo Ant-v5."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps (default: 1_000_000)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="Enable domain randomization (gravity, friction, mass)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        total_timesteps=args.timesteps,
        use_domain_randomization=args.domain_randomization,
        resume_from=args.resume,
    )
