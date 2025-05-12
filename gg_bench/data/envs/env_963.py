import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 0 (Add 1), 1 (Multiply by 2)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=1, high=23, shape=(1,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return (np.array([self.current_number], dtype=np.int32), 0, True, False, {})

        # Check if action is valid (action in {0,1})
        if action not in [0, 1]:
            # Invalid action index
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Determine valid actions
        valid_actions = []
        # Action 0: Add 1
        if self.current_number + 1 <= 23:
            valid_actions.append(0)
        # Action 1: Multiply by 2
        if self.current_number * 2 <= 23:
            valid_actions.append(1)

        if not valid_actions:
            # No valid moves available, current player loses
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        if action not in valid_actions:
            # Invalid move selected (would exceed 23)
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Apply the action
        if action == 0:
            self.current_number += 1
        elif action == 1:
            self.current_number *= 2

        # Check for winning condition
        if self.current_number == 23:
            self.done = True
            reward = 1
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch players
        self.current_player = 2 if self.current_player == 1 else 1
        return (np.array([self.current_number], dtype=np.int32), 0, False, False, {})

    def render(self):
        return f"Current Number: {self.current_number}, Player {self.current_player}'s turn."

    def valid_moves(self):
        valid_actions = []
        if self.current_number + 1 <= 23:
            valid_actions.append(0)
        if self.current_number * 2 <= 23:
            valid_actions.append(1)
        return valid_actions
