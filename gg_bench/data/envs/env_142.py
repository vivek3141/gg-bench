import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: multipliers from 2 to 9 (actions 0 to 7)
        self.action_space = spaces.Discrete(8)

        # Define observation space: current total (starting from 1)
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([1000000]), shape=(1,), dtype=np.int64
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_total = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_total], dtype=np.int64), {}

    def step(self, action):
        # Check if the action is valid
        if not self.action_space.contains(action):
            # Invalid action
            return (
                np.array([self.current_total], dtype=np.int64),
                -10,
                True,
                False,
                {},
            )

        # Map action to multiplier (actions 0-7 correspond to multipliers 2-9)
        multiplier = action + 2

        # Update the current total
        self.current_total *= multiplier

        # Check for winning condition
        if self.current_total >= 100:
            # Current player wins
            reward = 1
            self.done = True
            return (
                np.array([self.current_total], dtype=np.int64),
                reward,
                True,
                False,
                {},
            )
        else:
            # Valid move but the game continues
            reward = -10  # Penalty for a valid move
            # Switch to the next player
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
            return (
                np.array([self.current_total], dtype=np.int64),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        return f"Current total: {self.current_total}"

    def valid_moves(self):
        # All multipliers from 2 to 9 are always valid
        return list(range(8))
