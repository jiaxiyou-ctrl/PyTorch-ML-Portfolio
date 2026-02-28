"""Train PPO agent to control Ant from pixel observations."""

import os
import time

import gymnasium as gym
import numpy as np
import torch

from pixel_ppo_agent import PixelPPOAgent
from pixel_wrapper import PixelObsWrapper

CONFIG = {
    "image_size": 84,
    "frame_stack": 3,

    "total_timesteps": 50_000,
    "buffer_size": 2048,
    "batch_size": 64,
    "update_epochs": 4,
    "update_epoches":4,

    "lr_encoder": 1e-4,
    "lr_heads": 3e-4,

    "gamma": 0.99,
    "lam": 0.95,
    "clip_range": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,

    "use_augmentation": True,

    "log_interval": 1,
    "save_interval": 50,
    "save_dir": "checkpoints_pixel",
    
}

def make_pixel_env(image_size: int = 84, frame_stack: int = 3) -> gym.Env:
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

    print("=" * 60)
    print("Pixel PPO Training — Ant-v5")
    print("=" * 60)
    print(f"Image: {cfg['image_size']}x{cfg['image_size']}")
    print(f"Frame stack: {cfg['frame_stack']}")
    print(f"Buffer size: {cfg['buffer_size']}")
    print(f"Augmentation: {cfg['use_augmentation']}")
    print(f"Total timesteps: {cfg['total_timesteps']:,}")
    prunt(f"Interval: {cfg['log_interval']}")
    