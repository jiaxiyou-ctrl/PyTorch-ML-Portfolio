"""Gymnasium wrapper that replaces the default Ant reward with
language-conditioned task rewards (walk forward, turn left, etc.).
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

TASK_INSTRUCTIONS = {
    "walk_forward": {
        "text": "walk forward",
        "description": "Move forward along the positive X axis",
    },
    "walk_backward": {
        "text": "walk backward",
        "description": "Move backward along the negative X axis",
    },
    "turn_left": {
        "text": "turn left",
        "description": "Rotate counter-clockwise around the Z axis",
    },
    "turn_right": {
        "text": "turn right",
        "description": "Rotate clockwise around the Z axis",
    },
    "stand_still": {
        "text": "stand still",
        "description": "Stay in place without moving",
    },
}


class LanguageRewardWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        tasks: Optional[list] = None,
        healthy_reward: float = 1.0,
        task_reward_scale: float = 1.0,
    ) -> None:
        super().__init__(env)

        if tasks is None:
            self.task_names = list(TASK_INSTRUCTIONS.keys())
        else:
            self.task_names = tasks

        self.healthy_reward = healthy_reward
        self.task_reward_scale = task_reward_scale

        self.current_task: str = ""
        self.current_task_text: str = ""

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        obs, info = super().reset(**kwargs)
        self.current_task = np.random.choice(self.task_names)
        self.current_task_text = TASK_INSTRUCTIONS[self.current_task]["text"]

        info["task_name"] = self.current_task
        info["task_text"] = self.current_task_text

        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        obs, original_reward, terminated, truncated, info = super().step(action)

        data = self.env.unwrapped.data
        x_velocity = data.qvel[0]
        y_velocity = data.qvel[1]
        yaw_velocity = data.qvel[5]

        task_reward = self._compute_task_reward(                    
            x_velocity, y_velocity, yaw_velocity
        )

        custom_reward = self.healthy_reward + self.task_reward_scale * task_reward
        info["task_name"] = self.current_task
        info["task_text"] = self.current_task_text
        info["task_reward"] = task_reward
        info["original_reward"] = original_reward

        return obs, custom_reward, terminated, truncated, info

    def _compute_task_reward(
        self,
        x_velocity: float,
        y_velocity: float,
        yaw_velocity: float,
    ) -> float:

        if self.current_task == "walk_forward":
            return float(x_velocity)

        elif self.current_task == "walk_backward":
            return float(-x_velocity)

        elif self.current_task == "turn_left":
            return float(yaw_velocity)

        elif self.current_task == "turn_right":
            return float(-yaw_velocity)

        elif self.current_task == "stand_still":
            speed = abs(x_velocity) + abs(y_velocity) + 0.1 * abs(yaw_velocity)
            max_reward = 3.0
            return float(max_reward * np.exp(-speed))

        else:
            raise ValueError(f"Invalid task: {self.current_task}")
