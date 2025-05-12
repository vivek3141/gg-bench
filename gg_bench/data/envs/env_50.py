import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Double, 1 - Add One
        self.action_space = spaces.Discrete(2)

        # Observation space: Current number in the game
        # We set the high to 62 (31 * 2) to cover possible maximum after doubling
        self.observation_space = spaces.Box(low=1, high=62, shape=(1,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Can be 1 or 2
        self.done = False
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        if action not in [0, 1]:
            # Invalid action
            self.done = True
            return np.array([self.current_number], dtype=np.int32), -10, True, False, {}

        # Apply the chosen action
        if action == 0:
            # Double the current number
            self.current_number *= 2
        elif action == 1:
            # Add one to the current number
            self.current_number += 1

        # Check for victory or defeat
        if self.current_number == 31:
            # Current player wins
            self.done = True
            reward = 1
        elif self.current_number > 31:
            # Current player loses
            self.done = True
            reward = -10
        else:
            # Continue the game
            reward = 0
            # Switch to the other player
            self.current_player = 1 if self.current_player == 2 else 2

        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        return (
            f"Current Number: {self.current_number}\n"
            f"Player {self.current_player}'s turn."
        )

    def valid_moves(self):
        # Both actions are always valid in this game
        return [0, 1]
