import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.initial_N = 16  # Default starting number
        self.N_max = 100  # Maximum value for N
        self.action_space = spaces.Discrete(self.N_max + 1)  # Actions from 0 to N_max
        self.observation_space = spaces.Box(
            low=1, high=self.N_max, shape=(1,), dtype=np.int32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.initial_N
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def valid_moves(self):
        # Return all proper divisors of the current N
        return [i for i in range(1, self.N) if self.N % i == 0]

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}
        # Check if action is valid
        if action <= 0 or action >= self.N or self.N % action != 0:
            # Invalid move
            self.done = True
            reward = -10
        else:
            # Valid move
            self.N -= action
            # Switch player
            self.current_player *= -1
            # Check if current player has valid moves
            if not self.valid_moves():
                # Current player cannot make a move; previous player wins
                self.done = True
                reward = 1  # Previous player wins
            else:
                # Game continues
                reward = 0
        return np.array([self.N], dtype=np.int32), reward, self.done, False, {}

    def render(self):
        return (
            f"Current N: {self.N}, Player: {'1' if self.current_player == 1 else '2'}"
        )
