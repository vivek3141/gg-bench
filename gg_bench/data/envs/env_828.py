import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.starting_total = 23
        self.current_total = None
        self.current_player = None  # 1 or -1
        self.primes_list = [2, 3, 5, 7, 11, 13, 17, 19]
        self.action_space = spaces.Discrete(len(self.primes_list))
        self.observation_space = spaces.Box(
            low=0, high=self.starting_total, shape=(1,), dtype=np.int32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_total = self.starting_total
        self.current_player = 1  # Start with player 1
        self.done = False
        return np.array([self.current_total]), {}  # Observation and info

    def step(self, action):
        if self.done:
            return np.array([self.current_total]), 0, True, False, {}

        if action < 0 or action >= len(self.primes_list):
            # Invalid action index
            return np.array([self.current_total]), -10, True, False, {}

        prime = self.primes_list[action]

        if prime > self.current_total:
            # Invalid move
            return np.array([self.current_total]), -10, True, False, {}

        # Valid move
        self.current_total -= prime

        if self.current_total == 0:
            # Current player wins
            self.done = True
            return np.array([self.current_total]), 1, True, False, {}

        if self.current_total < 0:
            # Should not happen with valid moves
            self.done = True
            return np.array([self.current_total]), -10, True, False, {}

        # Switch players internally
        self.current_player *= -1
        return np.array([self.current_total]), 0, False, False, {}

    def render(self):
        return f"Current Total: {self.current_total}"

    def valid_moves(self):
        # Returns list of action indices for valid moves
        valid_actions = [
            i for i, p in enumerate(self.primes_list) if p <= self.current_total
        ]
        return valid_actions
