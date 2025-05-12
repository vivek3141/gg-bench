import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the maximum value for N (shared number)
        self.N_MAX = 1000

        # Starting number N
        self.N_STARTING_NUMBER = 16

        # Define action and observation space
        # The action_space is Discrete(N_MAX - 1), actions from 0 to N_MAX - 2
        # Each action corresponds to choosing a divisor: divisor = action + 2
        self.action_space = spaces.Discrete(self.N_MAX - 1)

        # Observation is the current value of N
        self.observation_space = spaces.Box(
            low=2, high=self.N_MAX, shape=(1,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_STARTING_NUMBER
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                np.array([self.N], dtype=np.int32),
                0,
                True,
                False,
                {},
            )  # Already done

        # Convert action index to actual divisor
        divisor = action + 2

        # Check if action is a valid proper divisor of N
        if divisor <= 1 or divisor >= self.N or self.N % divisor != 0:
            # Invalid move
            self.done = True
            return (
                np.array([self.N], dtype=np.int32),
                -10,  # Penalty for invalid move
                True,  # Episode ends
                False,
                {},
            )

        # Valid move; update N
        self.N = self.N // divisor

        # Check if N is prime; if so, current player wins
        if self.is_prime(self.N):
            self.done = True
            return (
                np.array([self.N], dtype=np.int32),
                1,  # Reward for winning
                True,  # Episode ends
                False,
                {},
            )

        # Game continues
        return np.array([self.N], dtype=np.int32), 0, False, False, {}

    def render(self):
        return f"Current N: {self.N}"

    def valid_moves(self):
        # Return a list of valid actions (action indices)
        # Valid actions correspond to proper divisors of N
        valid_actions = []
        for i in range(2, self.N):
            if self.N % i == 0:
                action = i - 2  # Convert divisor to action index
                valid_actions.append(action)
        return valid_actions

    def is_prime(self, n):
        if n <= 1:
            return False
        elif n <= 3:
            return True
        elif n % 2 == 0:
            return False
        sqrt_n = int(math.isqrt(n)) + 1
        for i in range(3, sqrt_n, 2):
            if n % i == 0:
                return False
        return True
