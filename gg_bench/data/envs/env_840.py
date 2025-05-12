import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers from 1 to 9 (mapped from indices 0 to 8)
        self.action_space = spaces.Discrete(9)

        # Define observation space: cumulative total from 0 to 27
        self.observation_space = spaces.Box(
            low=np.array([0]), high=np.array([27]), shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_total = 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.cumulative_total], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        # Map action index (0-8) to number (1-9)
        number = action + 1
        info = {}
        truncated = False

        # Check if action is valid
        if action < 0 or action >= 9:
            self.done = True
            reward = -10
            return (
                np.array([self.cumulative_total], dtype=np.int32),
                reward,
                self.done,
                truncated,
                info,
            )

        # Check if adding the number exceeds 27
        if self.cumulative_total + number > 27:
            self.done = True
            reward = -10
            return (
                np.array([self.cumulative_total], dtype=np.int32),
                reward,
                self.done,
                truncated,
                info,
            )

        # Add number to cumulative total
        self.cumulative_total += number

        # Check for win condition
        if self.cumulative_total == 27:
            self.done = True
            reward = 1
            return (
                np.array([self.cumulative_total], dtype=np.int32),
                reward,
                self.done,
                truncated,
                info,
            )

        # Switch players
        self.current_player *= -1  # Toggle between 1 and -1
        reward = 0
        return (
            np.array([self.cumulative_total], dtype=np.int32),
            reward,
            self.done,
            truncated,
            info,
        )

    def render(self):
        player = "Player 1" if self.current_player == 1 else "Player 2"
        return f"Cumulative total is now {self.cumulative_total}. {player}'s turn."

    def valid_moves(self):
        # Return a list of valid action indices (0-8 corresponding to numbers 1-9)
        valid_actions = [i for i in range(9) if self.cumulative_total + (i + 1) <= 27]
        return valid_actions
