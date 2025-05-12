import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum value for the initial N
        self.max_N = 100  # You can adjust this value as needed

        # Define action space: actions correspond to integers from 0 to max_N
        self.action_space = spaces.Discrete(self.max_N + 1)

        # Define observation space: current N
        self.observation_space = spaces.Box(
            low=np.array([2]), high=np.array([self.max_N]), shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.N = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Set initial N: choose a starting positive integer greater than 1
        self.N = 16  # Or any other initial value less than self.max_N
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, no further action can be taken
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        # Check if N is prime at the start of the turn
        if self.is_prime(self.N):
            # Current player cannot make a move
            self.done = True
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        # Check if action is valid
        if action <= 1 or action >= self.N or self.N % action != 0:
            # Invalid move
            self.done = True
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        # Valid move
        self.N = self.N // action

        # Check if after the move, N is prime
        if self.is_prime(self.N):
            # Current player wins
            self.done = True
            return np.array([self.N], dtype=np.int32), 1, True, False, {}
        else:
            # Continue the game
            return np.array([self.N], dtype=np.int32), 0, False, False, {}

    def render(self):
        return f"Current N is {self.N}"

    def valid_moves(self):
        # Return list of proper divisors of N
        if self.N <= 2:
            return []
        return [d for d in range(2, self.N) if self.N % d == 0]

    @staticmethod
    def is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
