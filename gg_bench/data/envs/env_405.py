import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to exponents 1, 2, or 3 (mapped from actions 0, 1, 2)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=2, high=np.inf, shape=(1,), dtype=np.float64
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.base_number = 2
        self.current_player = 1  # Player 1 starts (1 for Player 1, -1 for Player 2)
        self.done = False
        return np.array([self.base_number], dtype=np.float64), {}

    def step(self, action):
        if action not in self.valid_moves() or self.done:
            # Invalid action or game already finished
            return (
                np.array([self.base_number], dtype=np.float64),
                -10,
                True,
                False,
                {},
            )

        exponent = action + 1  # Map actions 0, 1, 2 to exponents 1, 2, 3
        new_base_number = self.base_number**exponent

        if new_base_number >= 256:
            # Current player wins
            self.base_number = new_base_number
            self.done = True
            return (
                np.array([self.base_number], dtype=np.float64),
                1,
                True,
                False,
                {},
            )
        else:
            # Continue game
            self.base_number = new_base_number
            self.current_player *= -1  # Switch player
            return (
                np.array([self.base_number], dtype=np.float64),
                0,
                False,
                False,
                {},
            )

    def render(self):
        return f"Player {1 if self.current_player == 1 else 2}'s turn.\nCurrent base number: {self.base_number}"

    def valid_moves(self):
        return [0, 1, 2]  # Actions corresponding to exponents 1, 2, 3
