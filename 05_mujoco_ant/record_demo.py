"""Record demo videos of the trained VLA-PPO agent."""

import os
import re
import torch
import imageio
from vla_train import make_vla_env
from vla_ppo_agent import VLAPPOAgent


def get_latest_checkpoint(checkpoint_dir):
    """Find the checkpoint with the highest iteration number."""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        return None

    # 🔧 FIX: 按数字排序，不是字母排序
    def extract_iter_num(filename):
        match = re.search(r'iter_(\d+)', filename)
        return int(match.group(1)) if match else 0

    checkpoints.sort(key=extract_iter_num)
    return os.path.join(checkpoint_dir, checkpoints[-1])


def record_task(agent, task_name, save_path, max_steps=500):
    """Record one episode of a specific task."""
    env = make_vla_env(image_size=84, frame_stack=3)
    obs, info = env.reset()

    # Force specific task
    env.current_task = task_name
    info["task_name"] = task_name

    frames = []
    total_reward = 0.0

    for step in range(max_steps):
        # Navigate through wrapper chain to get raw render
        raw_env = env
        while hasattr(raw_env, 'env'):
            raw_env = raw_env.env
        frame = raw_env.render()
        frames.append(frame)

        action, _, _ = agent.select_action(obs, task_name)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()

    # Save as MP4
    imageio.mimsave(save_path, frames, fps=30)
    print(f" {task_name}: reward={total_reward:.1f}, "
          f"steps={len(frames)}, saved to {save_path}")


def main():
    """Load checkpoint and record all 5 tasks."""
    # Setup agent
    env = make_vla_env()
    agent = VLAPPOAgent(
        obs_shape=env.observation_space.shape,
        act_dim=env.action_space.shape[0],
    )
    env.close()

    # Find latest checkpoint
    checkpoint_path = get_latest_checkpoint("checkpoints_vla_v4")
    if checkpoint_path is None:
        print(" No checkpoints found!")
        return

    # Load weights (weights_only=False for PyTorch 2.6+)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    agent.network.load_state_dict(checkpoint["network_state_dict"])

    print("=" * 60)
    print("VLA-PPO Demo Recording")
    print("=" * 60)
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Iteration:   {checkpoint['iteration']}")
    print(f"  Best reward: {checkpoint['best_reward']:.1f}")
    print("=" * 60)

    # Record each task
    os.makedirs("demo_videos", exist_ok=True)
    tasks = [
        "walk_forward",
        "walk_backward",
        "turn_left",
        "turn_right",
        "stand_still",
    ]

    print("\nRecording demos...")
    for task in tasks:
        save_path = f"demo_videos/{task}.mp4"
        record_task(agent, task, save_path)

    print("\n All demo videos saved in demo_videos/")
    print("   Open with: open demo_videos/")


if __name__ == "__main__":
    main()
