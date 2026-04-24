"""Smoke test: verify the VLA-PPO pipeline runs end-to-end without errors."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3_vla_ppo.vla_ppo_agent import VLAPPOAgent
from v3_vla_ppo.vla_train import make_vla_env

# 1. Test environment
env = make_vla_env(image_size=84, frame_stack=3)
obs, info = env.reset()
print(f" obs shape: {obs.shape}")
print(f" task: {info['task_name']}")

# 2. Test Agent
agent = VLAPPOAgent(
    obs_shape=obs.shape,
    act_dim=env.action_space.shape[0],
)
action, log_prob, value = agent.select_action(obs, info["task_name"])
print(f" action shape: {action.shape}")
print(f" log_prob: {log_prob:.4f}")
print(f" value: {value:.4f}")

# 3. Test environment interaction
next_obs, reward, terminated, truncated, info = env.step(action)
print(f" reward: {reward:.4f}")
print(f" next_obs shape: {next_obs.shape}")

print("\nAll smoke tests passed!")
env.close()
