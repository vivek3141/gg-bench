import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N_start=25):
        super(CustomEnv, self).__init__()
        self.N_start = N_start
        self.action_space = spaces.Discrete(9)  # Valid digits from 1 to 9
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(1,), dtype=np.int32
        )

        self.N = None
        self.current_player = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_start
        self.current_player = 1
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        digit = action + 1  # Map action index to digit (0 -> 1, ..., 8 -> 9)

        digits_in_N = [int(d) for d in str(self.N) if d != "0"]
        valid_digits = [d for d in digits_in_N if d <= self.N]

        if digit not in valid_digits:
            # Invalid move
            self.done = True
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        self.N -= digit

        if self.N == 0:
            # Current player wins
            self.done = True
            return np.array([self.N], dtype=np.int32), 1, True, False, {}
        else:
            # Valid move, switch player
            self.current_player *= -1
            return np.array([self.N], dtype=np.int32), -10, False, False, {}

    def render(self):
        return f"Current N: {self.N}, Current player: {self.current_player}"

    def valid_moves(self):
        if self.N <= 0:
            return []
        digits_in_N = [int(d) for d in str(self.N) if d != "0"]
        valid_digits = [d for d in digits_in_N if d <= self.N]
        valid_actions = [
            d - 1 for d in valid_digits
        ]  # Convert digits to action indices
        return valid_actions
