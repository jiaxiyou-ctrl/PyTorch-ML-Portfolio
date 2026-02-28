import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("Ant-v5", render_mode="rgb_array")

obs, info = env.reset()

image = env.render()


print(f"Observation shape: {obs.shape}")
print(f"Image shape: {image.shape}")
print(f"{image.min()} - {image.max()}")

plt.imshow(image)
plt.title("Ant's first view")
plt.axis("off")
plt.savefig("ant_first_view.png")
plt.show()
env.close()
