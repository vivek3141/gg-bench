import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space consists of integers 0-8 corresponding to digits 1-9
        self.action_space = spaces.Discrete(9)  # Actions 0-8 correspond to digits 1-9
        # The observation is the current shared number, an integer between 0 and 100
        self.observation_space = spaces.Box(
            low=np.array([0]), high=np.array([100]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = 100
        self.done = False
        return np.array([self.shared_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.shared_number], dtype=np.int32), 0, True, False, {}

        # Map action index to digit (1-9)
        digit = action + 1

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return (
                np.array([self.shared_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        self.shared_number -= digit

        if self.shared_number < 0:
            # This shouldn't happen due to the move validation, but handle just in case
            self.done = True
            reward = -10
            return (
                np.array([self.shared_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        if self.shared_number == 0:
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.shared_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Continue the game
        reward = 0
        return np.array([self.shared_number], dtype=np.int32), reward, False, False, {}

    def render(self):
        return f"Current Shared Number: {self.shared_number}"

    def valid_moves(self):
        digits = [int(d) for d in str(self.shared_number)]
        valid_digits = [d for d in digits if d != 0]
        # Map digits to action indices (digit - 1)
        valid_actions = [d - 1 for d in valid_digits]
        return valid_actions
