import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from pixel_wrapper import PixelObsWrapper

def test_wrapper():

    raw_env = gym.make("Ant-v5", render_mode="rgb_array")
    env = PixelObsWrapper(raw_env, image_size=84, frame_stack=3)

    print(f"Raw observation space: {raw_env.observation_space}")
    print(f"Pixel observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"[{obs.min()}, {obs.max()}]")

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    print(f"Observation shape: {obs2.shape}")
    print(f"[{obs2.min()}, {obs2.max()}]")

    frame1 = obs2[0:3].transpose(1, 2, 0)
    frame2 = obs2[3:6].transpose(1, 2, 0)
    frame3 = obs2[6:9].transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax,frame, title in zip(axes, [frame1, frame2, frame3], ["Frame 1(oldest)", "Frame 2", "Frame 3(newest)"]):
        ax.imshow(frame)
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle("Staked Frames from PixelObsWrapper")
    plt.tight_layout()
    plt.savefig("pixel_obs_frames.png")
    plt.show()


    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()

if __name__ == "__main__":
    test_wrapper()