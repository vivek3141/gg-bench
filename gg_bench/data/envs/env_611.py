import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(8) corresponding to multipliers 2-9
        self.action_space = spaces.Discrete(8)
        # Observation space: cumulative product as a scalar in a Box
        self.observation_space = spaces.Box(
            low=1.0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_product = 1.0
        self.current_player = 1  # Player 1 or Player -1
        self.done = False
        return (
            np.array([self.cumulative_product], dtype=np.float32),
            {},
        )  # Observation and info

    def step(self, action):
        if action not in self.valid_moves() or self.done:
            # Invalid move or game already over
            return (
                np.array([self.cumulative_product], dtype=np.float32),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Apply the action
        multiplier = action + 2  # Actions 0-7 correspond to multipliers 2-9
        self.cumulative_product *= multiplier

        if self.cumulative_product >= 1000:
            # Current player wins
            reward = 1
            self.done = True
            return (
                np.array([self.cumulative_product], dtype=np.float32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Valid move, switch player
            reward = -10
            self.current_player *= -1  # Switch player
            return (
                np.array([self.cumulative_product], dtype=np.float32),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        return (
            f"Cumulative product is {self.cumulative_product}\n"
            f"Player {'1' if self.current_player == 1 else '2'}'s turn."
        )

    def valid_moves(self):
        # Valid moves are always from 0 to 7, unless the game is over
        if self.done:
            return []
        else:
            return list(range(8))
