import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=100):
        super(CustomEnv, self).__init__()

        self.starting_number = starting_number
        self.N_MAX = self.starting_number

        # Define action_space: actions are integers from 0 to N_MAX
        # Actions correspond to possible divisors to subtract
        self.action_space = spaces.Discrete(self.N_MAX + 1)  # Actions from 0 to N_MAX

        # Define observation_space: current N ranges from 1 to N_MAX
        self.observation_space = spaces.Box(
            low=1, high=self.N_MAX, shape=(1,), dtype=int
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.starting_number
        self.done = False
        return np.array([self.N]), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return np.array([self.N]), 0, True, False, {}  # Game is already over

        divisor = action

        # Check if action is valid
        if self.is_valid_divisor(divisor):
            # Valid move
            self.N -= divisor

            # Check if opponent has any legal moves
            valid_divisors = self.get_proper_divisors(self.N)
            if len(valid_divisors) == 0:
                # Opponent cannot move, current player wins
                self.done = True
                reward = 1  # Current player wins
                return np.array([self.N]), reward, True, False, {}
            else:
                # Game continues
                reward = -10  # Penalty for valid move
                return np.array([self.N]), reward, False, False, {}
        else:
            # Invalid move, current player loses
            self.done = True
            reward = -10  # Penalty for invalid move
            return np.array([self.N]), reward, True, False, {}

    def is_valid_divisor(self, divisor):
        # Check if the divisor is a proper divisor of N
        if divisor <= 1 or divisor >= self.N:
            return False
        if self.N % divisor != 0:
            return False
        return True

    def get_proper_divisors(self, N):
        # Return a list of proper divisors of N
        divisors = []
        for i in range(2, N):
            if N % i == 0:
                divisors.append(i)
        return divisors

    def valid_moves(self):
        # Return a list of valid moves (proper divisors of N)
        return self.get_proper_divisors(self.N)

    def render(self):
        return f"Current N: {self.N}, Proper divisors: {self.valid_moves()}"
