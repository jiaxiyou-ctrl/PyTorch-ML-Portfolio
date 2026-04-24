"""Train PPO agent to control Ant from pixel observations."""

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import gymnasium as gym
import numpy as np
import torch

from v2_pixel_ppo.pixel_ppo_agent import PixelPPOAgent
from v2_pixel_ppo.pixel_wrapper import PixelObsWrapper

CONFIG = {
    "image_size": 84,
    "frame_stack": 3,

    "total_timesteps": 5_000_000,
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
    "save_interval": 50,
    "save_dir": "checkpoints_pixel",
    
}

def make_pixel_env(image_size: int = 84, frame_stack: int = 3) -> gym.Env:
    """Create Ant-v5 wrapped with pixel observations and frame stacking."""
    raw_env = gym.make("Ant-v5", render_mode="rgb_array")
    pixel_env = PixelObsWrapper(raw_env, image_size=image_size, frame_stack=frame_stack)
    return pixel_env

def save_checkpoint(
    agent: PixelPPOAgent,
    save_dir: str,
    iteration: int,
    global_step: int,
    best_reward: float,   
) -> None:
    """Serialise network + optimizer state to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"pixel_ppo_iter_{iteration}.pt")

    torch.save(
        {
            "iteration": iteration,
            "global_step": global_step,
            "best_reward": best_reward,
            "network_state_dict": agent.network.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
        },
        path,
    )
    print(f"   ->  Checkpoint saved: {path}")

def train() -> None:
    """Main training function."""

    cfg = CONFIG

    # ── Config printout ──────────────────────────────────────────
    print("=" * 60)
    print("Pixel-based PPO Training for Ant-v5")
    print("=" * 60)
    print(f"  Image:        {cfg['image_size']}×{cfg['image_size']}")
    print(f"  Frame stack:  {cfg['frame_stack']}")
    print(f"  Buffer size:  {cfg['buffer_size']}")
    print(f"  Augmentation: {cfg['use_augmentation']}")
    print(f"  Total steps:  {cfg['total_timesteps']:,}")
    print(
        f"  Iterations:   ~{cfg['total_timesteps'] // cfg['buffer_size']}"
    )
    print("=" * 60)

    # ── Environment and agent initialisation ───────────────────
    env = make_pixel_env(cfg['image_size'], cfg['frame_stack'])
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    agent = PixelPPOAgent(
        obs_shape=obs_shape,
        act_dim=act_dim,
        lr_encoder=cfg['lr_encoder'],
        lr_heads=cfg['lr_heads'],
        gamma=cfg['gamma'],
        lam=cfg['lam'],
        clip_range=cfg['clip_range'],
        entropy_coef=cfg['entropy_coef'],
        value_coef=cfg['value_coef'],
        max_grad_norm=cfg['max_grad_norm'],
        update_epochs=cfg['update_epochs'],
        batch_size=cfg['batch_size'],
        buffer_size=cfg['buffer_size'],
        use_augmentation=cfg['use_augmentation'],
        normalize_reward=cfg['normalize_reward'],
    )

    total_params = sum(p.numel() for p in agent.network.parameters())

    global_step = 0
    iteration = 0
    best_reward = -float('inf')

    episode_reward = 0.0
    episode_length = 0
    episode_rewards =[]

    obs, _ = env.reset()
    train_start_time = time.time()

    print(f"   ->  Training started!")

    while global_step < cfg['total_timesteps']:
        iteration += 1
        iter_start_time = time.time()

        # ── Rollout collection ───────────────────────────────────
        for _step in range(cfg['buffer_size']):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            normalized_reward = agent.normalize_rew(reward)
            agent.buffer.store(obs, action, normalized_reward, value, log_prob, float(done))
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

        # ── Advantage computation ─────────────────────────────────
        with torch.no_grad():
            last_obs_tensor = torch.tensor(obs, dtype=torch.float32)
            last_value = agent.network.get_value(last_obs_tensor).item()

        agent.buffer.compute_advantages(last_value, gamma=cfg["gamma"], lam=cfg["lam"])

        # ── Policy update ────────────────────────────────────────
        agent.update()

        iter_time = time.time() - iter_start_time
        total_time = time.time() - train_start_time

        # ── Logging ──────────────────────────────────────────────
        if iteration % cfg['log_interval'] == 0 and len(episode_rewards) > 0:
            recent = episode_rewards[-10:]
            avg_reward = np.mean(recent)
            min_reward = np.min(recent)
            max_reward = np.max(recent)
            
            steps_per_sec = cfg["buffer_size"] / iter_time

            print(
                f"Iter {iteration:4d} | "
                f"Step {global_step:>8,} | "
                f"Reward {avg_reward:7.1f} | "
                f"min={min_reward:.0f}, max={max_reward:.0f} | "
                f"Episodes {len(episode_rewards):4d} | "
                f"Speed: {steps_per_sec:.0f} steps/s | "
                f"Time {total_time /60:.1f} min"
            )

            if avg_reward > best_reward:
                best_reward = avg_reward
                
        if iteration % cfg['save_interval'] == 0:
            save_checkpoint(
                agent, cfg['save_dir'], iteration, global_step, best_reward
            )

    total_time = time.time() - train_start_time
    print(f"Training complete!")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Total steps: {global_step:,}")
    print(f"Total episodes: {len(episode_rewards):,}")
    print(f"Total time: {total_time / 60:.1f} min")
    print(f"Total params: {total_params:,}")


    save_checkpoint(
        agent, cfg['save_dir'], iteration, global_step, best_reward
    )
    env.close()

if __name__ == "__main__":
    train()
  