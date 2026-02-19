"""
train.py
--------
Training script for the Q-Learning maze solver.

Runs the agent for a configurable number of episodes, then:
  1. Saves a reward-curve plot  → results/training_reward_curve.png
  2. Saves an animated GIF of the trained agent navigating the maze
                               → results/maze_agent.gif

Usage:
    python train.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from maze_env import SimpleMaze
from q_learning_agent import QLearningAgent

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_EPISODES = 1000
MAX_STEPS = 200
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
LOG_INTERVAL = 100

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def draw_maze_frame(ax, grid, step_num, episode_reward):
    """Render one maze frame onto *ax* using a colour-coded grid."""
    ax.clear()

    # Colour map: 0=floor, 1=wall, 2=agent, 3=goal
    color_grid = np.zeros((*grid.shape, 3))
    color_map = {
        0: [0.95, 0.95, 0.95],   # floor  – light grey
        1: [0.20, 0.20, 0.20],   # wall   – dark grey
        2: [0.20, 0.60, 0.90],   # agent  – blue
        3: [1.00, 0.80, 0.10],   # goal   – yellow
    }
    for val, color in color_map.items():
        color_grid[grid == val] = color

    ax.imshow(color_grid, interpolation="nearest")

    size = grid.shape[0]
    for r in range(size):
        for c in range(size):
            cell = grid[r, c]
            if cell == 2:
                ax.text(c, r, "🐭", ha="center", va="center", fontsize=14)
            elif cell == 3:
                ax.text(c, r, "🧀", ha="center", va="center", fontsize=14)
            elif cell == 1:
                ax.text(c, r, "✖", ha="center", va="center",
                        fontsize=12, color="white")

    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_title(f"Step {step_num}  |  Reward so far: {episode_reward}",
                 fontsize=10)

    legend_elements = [
        mpatches.Patch(facecolor=[0.20, 0.60, 0.90], label="Agent (mouse)"),
        mpatches.Patch(facecolor=[1.00, 0.80, 0.10], label="Goal (cheese)"),
        mpatches.Patch(facecolor=[0.20, 0.20, 0.20], label="Wall"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=7, framealpha=0.8)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    """Train the Q-Learning agent and return per-episode rewards."""
    env = SimpleMaze()
    agent = QLearningAgent(
        state_size=env.size * env.size,
        num_actions=len(env.actions),
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
    )

    episode_rewards = []

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if episode % LOG_INTERVAL == 0:
            avg = np.mean(episode_rewards[-LOG_INTERVAL:])
            print(
                f"Episode {episode:>5}/{NUM_EPISODES}"
                f" | Reward: {total_reward:>6.1f}"
                f" | Avg({LOG_INTERVAL}): {avg:>6.1f}"
                f" | Epsilon: {agent.epsilon:.3f}"
                f" | Steps: {step + 1}"
            )

    return agent, episode_rewards


# ---------------------------------------------------------------------------
# Visualisation 1: reward curve
# ---------------------------------------------------------------------------

def plot_reward_curve(episode_rewards, save_path):
    """Plot and save the training reward curve."""
    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(1, len(episode_rewards) + 1)
    ax.plot(episodes, episode_rewards, alpha=0.35, color="steelblue",
            linewidth=0.8, label="Episode reward")

    # Smoothed moving average (window = 50)
    window = 50
    smoothed = np.convolve(episode_rewards,
                           np.ones(window) / window, mode="valid")
    ax.plot(np.arange(window, len(episode_rewards) + 1), smoothed,
            color="darkorange", linewidth=2,
            label=f"Moving avg (window={window})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("Q-Learning Maze Solver — Training Reward Curve", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Reward curve saved → {save_path}")


# ---------------------------------------------------------------------------
# Visualisation 2: animated GIF of the trained agent
# ---------------------------------------------------------------------------

def record_episode(agent, env, max_steps=MAX_STEPS):
    """
    Run one greedy episode (epsilon=0) and collect frame data.

    Returns:
        list[tuple]: Each element is (grid_snapshot, cumulative_reward).
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0   # pure exploitation

    state = env.reset()
    frames = [(env.render_to_grid().copy(), 0)]
    cumulative = 0

    for _ in range(max_steps):
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        cumulative += reward
        frames.append((env.render_to_grid().copy(), cumulative))
        if done:
            break

    agent.epsilon = original_epsilon
    return frames


def save_agent_gif(agent, env, save_path):
    """Animate a single greedy episode and save as an animated GIF."""
    frames = record_episode(agent, env)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("white")

    def update(frame_idx):
        grid, cum_reward = frames[frame_idx]
        draw_maze_frame(ax, grid, frame_idx, cum_reward)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=400,     # ms per frame
        repeat=False,
    )

    ani.save(save_path, writer="pillow", fps=2.5)
    plt.close(fig)
    print(f"Agent GIF saved       → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Q-Learning Maze Solver — Training")
    print("=" * 60)

    trained_agent, rewards = train()

    print("\nGenerating visualisations...")
    reward_curve_path = os.path.join(RESULTS_DIR, "training_reward_curve.png")
    gif_path = os.path.join(RESULTS_DIR, "maze_agent.gif")

    plot_reward_curve(rewards, reward_curve_path)

    demo_env = SimpleMaze()
    save_agent_gif(trained_agent, demo_env, gif_path)

    print("\nDone! Check the results/ folder for output files.")
