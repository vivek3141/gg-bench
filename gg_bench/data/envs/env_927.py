import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum value of N
        self.max_N = 100

        # Define action and observation space
        # The action is the proper divisor to subtract
        self.action_space = spaces.Discrete(
            self.max_N + 1
        )  # Actions from 0 to max_N inclusive

        # Observation is the current value of N
        self.observation_space = spaces.Box(
            low=1, high=self.max_N, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.N = None
        self.current_player = 1
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize N to a composite number greater than 30
        self.N = self.np_random.integers(31, self.max_N + 1)
        while self.is_prime(self.N):
            self.N = self.np_random.integers(31, self.max_N + 1)

        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        # Check if action is valid
        if action >= 2 and action < self.N and self.N % action == 0:
            # Valid move
            self.N -= action

            # Check if the next player can move
            if not self.has_proper_divisors(self.N):
                self.done = True
                return np.array([self.N], dtype=np.int32), 1, True, False, {}

            # Swap players
            self.current_player *= -1
            return np.array([self.N], dtype=np.int32), 0, False, False, {}

        else:
            # Invalid move
            self.done = True
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

    def render(self):
        return f"Current N: {self.N}, Player {'1' if self.current_player == 1 else '2'}'s turn."

    def valid_moves(self):
        return self.proper_divisors(self.N)

    def proper_divisors(self, n):
        return [d for d in range(2, n) if n % d == 0]

    def has_proper_divisors(self, n):
        for d in range(2, n):
            if n % d == 0:
                return True
        return False

    def is_prime(self, n):
        if n <= 3:
            return n > 1
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        w = 2
        while i * i <= n:
            if n % i == 0:
                return False
            i += w
            w = 6 - w
        return True
