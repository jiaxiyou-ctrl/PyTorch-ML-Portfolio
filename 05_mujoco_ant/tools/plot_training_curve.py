"""Plot training reward curve from a training log file."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path: str) -> Tuple[List[int], List[float]]:
    """Extract (step, mean_reward) pairs from train.py log output."""
    steps: List[int] = []
    rewards: List[float] = []

    with open(log_path, "r") as f:
        for line in f:
            match = re.search(
                r"Step\s+([\d,]+)\s+.*Mean reward\s+([-\d.]+)", line
            )
            if match:
                steps.append(int(match.group(1).replace(",", "")))
                rewards.append(float(match.group(2)))

    return steps, rewards


def smooth(values: List[float], weight: float = 0.9) -> List[float]:
    """Exponential moving average."""
    smoothed: List[float] = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def plot(steps, rewards, save_path="results/training_reward_curve.png"):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        steps, rewards,
        color="#2196F3", alpha=0.2, linewidth=0.8,
        label="Raw",
    )

    smoothed = smooth(rewards, weight=0.9)
    ax.plot(
        steps, smoothed,
        color="#2196F3", linewidth=2.5,
        label="Smoothed (EMA 0.9)",
    )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    best_idx = int(np.argmax(rewards))
    ax.scatter(
        steps[best_idx], rewards[best_idx],
        color="#FF5722", s=80, zorder=5,
        label=f"Best: {rewards[best_idx]:+.1f} (step {steps[best_idx]:,})",
    )

    # Mark where smoothed reward first crosses zero
    for i in range(1, len(rewards)):
        if smoothed[i - 1] < 0 and smoothed[i] >= 0:
            ax.axvline(
                x=steps[i], color="#4CAF50",
                linestyle=":", linewidth=1.5, alpha=0.7,
            )
            ax.annotate(
                f"Zero crossing\nstep {steps[i]:,}",
                xy=(steps[i], 0),
                xytext=(steps[i] + 200_000, -100),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#4CAF50"),
                color="#4CAF50",
            )
            break

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("PPO Training on Ant-v5", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M")
    )

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curve from log.")
    parser.add_argument("--log", default="training_log.txt")
    parser.add_argument("--output", default="results/training_reward_curve.png")
    args = parser.parse_args()

    steps, rewards = parse_log(args.log)

    if len(steps) == 0:
        print(f"No data found in {args.log}")
        print("Expected format: Update .../... | Step ... | ... | Mean reward ... | ...")
    else:
        print(f"Found {len(steps)} data points")
        print(f"  Steps: {steps[0]:,} -> {steps[-1]:,}")
        print(f"  Reward: {min(rewards):.1f} -> {max(rewards):.1f}")
        plot(steps, rewards, save_path=args.output)
