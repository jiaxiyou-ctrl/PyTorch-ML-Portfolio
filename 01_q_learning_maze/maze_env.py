"""
maze_env.py
-----------
A simple 4x4 grid-world maze environment for reinforcement learning experiments.

Maze layout:
    ┌───┬───┬───┬───┐
    │ S │   │   │   │     S = Start (0, 0)
    ├───┼───┼───┼───┤
    │   │ X │   │   │     X = Wall (impassable)
    ├───┼───┼───┼───┤
    │   │   │ X │   │     G = Goal / Cheese (3, 3)
    ├───┼───┼───┼───┤
    │   │   │   │ G │
    └───┴───┴───┴───┘
"""

import numpy as np


class SimpleMaze:
    """
    A simple 4x4 maze environment.

    The agent (mouse) starts at (0, 0) and must reach the goal (cheese) at (3, 3).
    Two walls block direct paths, requiring the agent to navigate around them.

    Rewards:
        +100  : reaching the goal
        -1    : each valid step (time penalty to encourage efficiency)
        -5    : hitting a wall or stepping out of bounds (position unchanged)
    """

    def __init__(self):
        self.size = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.walls = [(1, 1), (2, 2)]
        self.state = self.start

        # Action index -> (delta_row, delta_col)
        self.actions = {
            0: (-1, 0),   # up:    row - 1
            1: (1, 0),    # down:  row + 1
            2: (0, -1),   # left:  col - 1
            3: (0, 1),    # right: col + 1
        }
        self.action_names = ['up', 'down', 'left', 'right']

    def reset(self):
        """Reset the agent to the start position and return the initial state."""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Execute an action and return (new_state, reward, done).

        Args:
            action (int): Action index (0=up, 1=down, 2=left, 3=right).

        Returns:
            tuple: (new_state, reward, done)
        """
        row, col = self.state
        d_row, d_col = self.actions[action]
        new_row = row + d_row
        new_col = col + d_col
        new_state = (new_row, new_col)

        # Check for out-of-bounds or wall collision — stay in place
        if (new_row < 0 or new_row >= self.size or
                new_col < 0 or new_col >= self.size or
                new_state in self.walls):
            return self.state, -5, False

        self.state = new_state

        if new_state == self.goal:
            return new_state, 100, True

        return new_state, -1, False

    def render(self):
        """Print a text representation of the current maze state to stdout."""
        for r in range(self.size):
            row_str = ""
            for c in range(self.size):
                if (r, c) == self.state:
                    row_str += "🐭"
                elif (r, c) == self.goal:
                    row_str += "🧀"
                elif (r, c) in self.walls:
                    row_str += "██"
                else:
                    row_str += ". "
            print(row_str)
        print()

    def render_to_grid(self):
        """
        Return the maze as a 2D numpy array for visualization purposes.

        Cell values:
            0 = empty, 1 = wall, 2 = agent, 3 = goal
        """
        grid = np.zeros((self.size, self.size), dtype=int)
        for wall in self.walls:
            grid[wall] = 1
        grid[self.goal] = 3
        grid[self.state] = 2
        return grid
