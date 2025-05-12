import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: Discrete(8) for multipliers 2 to 9
        self.action_space = spaces.Discrete(
            8
        )  # Actions 0 to 7 correspond to multipliers 2 to 9

        # Define observation space: cumulative product as a single integer value
        # We set the high to 1000 assuming the product won't exceed this in valid gameplay
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([1000]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_product = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.cumulative_product], dtype=np.int32),
            {},
        )  # observation, info

    def step(self, action):
        if self.done:
            # If the game is already over, return the current state
            return (
                np.array([self.cumulative_product], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        valid_actions = self.valid_moves()

        if len(valid_actions) == 0:
            # No valid moves available; current player loses
            self.done = True
            reward = -10  # Current player loses
            return (
                np.array([self.cumulative_product], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        if action not in valid_actions:
            # Invalid move; current player loses
            self.done = True
            reward = -10  # Current player loses
            return (
                np.array([self.cumulative_product], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        multiplier = action + 2  # Map action index to multiplier (2-9)
        self.cumulative_product *= multiplier

        if self.cumulative_product == 100:
            # Current player wins
            self.done = True
            reward = 1  # Winning reward
            return (
                np.array([self.cumulative_product], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        elif self.cumulative_product > 100:
            # Should not occur if valid moves are correctly checked, but handle just in case
            self.done = True
            reward = -10  # Current player loses
            return (
                np.array([self.cumulative_product], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        else:
            # Game continues; switch to the next player
            self.current_player = 1 if self.current_player == 2 else 2
            return (
                np.array([self.cumulative_product], dtype=np.int32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        # Return a string representation of the game state
        return f"Cumulative Product: {self.cumulative_product}, Player {self.current_player}'s turn"

    def valid_moves(self):
        # List of valid action indices (0-7) corresponding to multipliers that do not exceed 100
        valid_actions = []
        for action in range(8):  # Actions 0 to 7
            multiplier = action + 2  # Map action index to multiplier (2-9)
            if self.cumulative_product * multiplier <= 100:
                valid_actions.append(action)
        return valid_actions
