"""
q_learning_agent.py
-------------------
A tabular Q-Learning agent that learns optimal policies via the Bellman equation.

The Q-table maps (state, action) pairs to expected cumulative rewards.
Exploration is controlled by an epsilon-greedy strategy that decays over time.
"""

import random


class QLearningAgent:
    """
    Tabular Q-Learning agent with epsilon-greedy exploration.

    The Q-table is stored as a dictionary: {state: [q_value_per_action]}.
    States are added lazily on first visit, initialised to zero.

    Args:
        state_size  : Unused (kept for API clarity; table grows dynamically).
        num_actions : Number of discrete actions available.
        learning_rate   : Step size alpha for Q-value updates (default 0.1).
        discount_factor : Future reward discount gamma (default 0.99).
        epsilon         : Initial exploration probability (default 1.0).
        epsilon_decay   : Multiplicative decay applied after each episode (default 0.995).
        epsilon_min     : Lower bound for epsilon (default 0.01).
    """

    def __init__(
        self,
        state_size,
        num_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_values(self, state):
        """
        Return the Q-value list for a given state, initialising to zeros if unseen.

        Args:
            state: Hashable state identifier (e.g. a tuple).

        Returns:
            list[float]: Q-values for each action.
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_actions
        return self.q_table[state]

    def choose_action(self, state):
        """
        Select an action using the epsilon-greedy policy.

        With probability epsilon the agent explores (random action);
        otherwise it exploits the current best known action.

        Args:
            state: Current environment state.

        Returns:
            int: Chosen action index.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        q_values = self.get_q_values(state)
        return q_values.index(max(q_values))

    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table using the Bellman equation.

        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

        Args:
            state      : State before the action.
            action     : Action taken.
            reward     : Reward received.
            next_state : State reached after the action.
            done       : Whether the episode ended.
        """
        q_values = self.get_q_values(state)
        old_q = q_values[action]

        if done:
            target = reward
        else:
            next_q_values = self.get_q_values(next_state)
            target = reward + self.gamma * max(next_q_values)

        # Nudge Q-value toward the target
        q_values[action] = old_q + self.lr * (target - old_q)

    def decay_epsilon(self):
        """Reduce exploration probability after each episode, down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
