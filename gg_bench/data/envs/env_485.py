import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_N=30):
        super(CustomEnv, self).__init__()

        self.starting_N = starting_N

        # Define action space: actions correspond to potential divisors D
        # We allow actions from 0 to starting_N (inclusive)
        self.action_space = spaces.Discrete(self.starting_N + 1)

        # Define observation space: the current value of N
        self.observation_space = spaces.Box(
            low=1, high=self.starting_N, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.starting_N
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N]), {}  # Return observation and info

    def step(self, action):
        D = action  # The chosen divisor
        reward = 0
        info = {}

        if self.done:
            # Game is already over
            return np.array([self.N]), 0, True, False, info

        # Check if the action is a valid proper divisor
        if D <= 1 or D >= self.N or self.N % D != 0:
            # Invalid move
            self.done = True
            reward = -10
            return np.array([self.N]), reward, True, False, info

        # Valid move: divide N by D
        self.N = self.N // D

        if self.N == 1:
            # Current player wins
            self.done = True
            reward = 1
            return np.array([self.N]), reward, True, False, info

        if self.N > 2 and self.is_prime(self.N):
            # Next player cannot move; current player wins
            self.done = True
            reward = 1
            return np.array([self.N]), reward, True, False, info

        # Switch players
        self.current_player = 3 - self.current_player

        return np.array([self.N]), reward, False, False, info

    def render(self):
        return f"Current N: {self.N}\n"

    def valid_moves(self):
        # Return a list of valid action indices
        return [D for D in range(2, self.N) if self.N % D == 0]

    def is_prime(self, n):
        if n <= 1:
            return False
        elif n <= 3:
            return True
        elif n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True
