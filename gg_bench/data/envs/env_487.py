import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Define action and observation space
        # Action space: actions from 0 to 98, representing divisors from 2 to 100
        self.action_space = spaces.Discrete(99)

        # Observation space: the Shared Number, an integer between 1 and 100
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([100]), shape=(1,), dtype=np.int32
        )

        # Initialize the Shared Number and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.SharedNumber = 100
        self.done = False
        # Return observation and info
        return np.array([self.SharedNumber], dtype=np.int32), {}

    def step(self, action):
        # Map action index to divisor
        divisor = action + 2  # action 0 corresponds to divisor 2

        if self.done:
            # If the game is already done, return current state with zero reward
            return np.array([self.SharedNumber], dtype=np.int32), 0, True, False, {}

        # Check if action is valid
        if (
            divisor >= 2
            and divisor < self.SharedNumber
            and self.SharedNumber % divisor == 0
        ):
            # Valid move
            self.SharedNumber //= divisor
            # Check winning conditions
            if self.SharedNumber == 1:
                # Current player wins
                self.done = True
                return np.array([self.SharedNumber], dtype=np.int32), 1, True, False, {}
            elif self.is_prime(self.SharedNumber):
                # Next player cannot move, current player wins
                self.done = True
                return np.array([self.SharedNumber], dtype=np.int32), 1, True, False, {}
            else:
                # Game continues
                return (
                    np.array([self.SharedNumber], dtype=np.int32),
                    -10,
                    False,
                    False,
                    {},
                )
        else:
            # Invalid move, current player loses
            self.done = True
            return np.array([self.SharedNumber], dtype=np.int32), -10, True, False, {}

    def render(self):
        # Return a visual representation of the environment state as a string
        return f"Current Shared Number: {self.SharedNumber}"

    def valid_moves(self):
        if self.SharedNumber <= 1:
            return []
        return [
            a
            for a in range(99)
            if (a + 2) < self.SharedNumber and self.SharedNumber % (a + 2) == 0
        ]

    def is_prime(self, n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0:
            return n == 2
        sqrt_n = int(np.sqrt(n)) + 1
        for i in range(3, sqrt_n, 2):
            if n % i == 0:
                return False
        return True
