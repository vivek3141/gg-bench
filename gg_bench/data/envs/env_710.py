import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.N_start = 100  # Starting N
        self.N = self.N_start  # Current value of N
        self.current_player = 1  # 1 or 2

        # Define action and observation spaces
        self.action_space = spaces.Discrete(
            self.N_start + 1
        )  # Actions from 0 to N_start
        self.observation_space = spaces.Box(
            low=1, high=self.N_start, shape=(1,), dtype=np.int64
        )
        self.done = False  # Indicates if the game is over

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_start
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N]), {}  # Return observation and info

    def get_proper_divisors(self, N):
        """Helper function to get proper divisors of N greater than 1."""
        return [i for i in range(2, N) if N % i == 0]

    def valid_moves(self):
        """Return a list of valid moves (proper divisors of N)."""
        return self.get_proper_divisors(self.N)

    def step(self, action):
        if self.done:
            return np.array([self.N]), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move, current player loses
            reward = -10
            self.done = True
            return np.array([self.N]), reward, True, False, {}

        # Valid move, update N
        self.N = self.N // action

        # Check for winning condition
        if self.N == 1 or len(self.valid_moves()) == 0:
            # Current player wins
            reward = 1
            self.done = True
            return np.array([self.N]), reward, True, False, {}
        else:
            # Switch player
            self.current_player = 3 - self.current_player  # Switches between 1 and 2
            reward = -10  # Reward for a valid move
            return np.array([self.N]), reward, False, False, {}

    def render(self):
        """Return a string representation of the current game state."""
        return (
            f"Current N: {self.N}\n"
            f"Current Player: Player {self.current_player}\n"
            f"Proper Divisors: {self.valid_moves()}"
        )
