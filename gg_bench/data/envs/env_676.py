import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the maximum possible value for N
        self.N_max = 1000

        # Define action and observation space
        # The action is the chosen proper divisor (integer between 0 and N_max)
        self.action_space = spaces.Discrete(self.N_max + 1)

        # The observation is the current number N
        self.observation_space = spaces.Box(
            low=1, high=self.N_max, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_N = 100  # Starting number N
        self.done = False
        return np.array([self.current_N], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        if self.done:
            # Game has already ended
            return np.array([self.current_N], dtype=np.int32), -10, True, False, {}

        # Get the list of valid moves
        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move; the player loses
            self.done = True
            return np.array([self.current_N], dtype=np.int32), -10, True, False, {}

        # Apply the action: subtract the chosen proper divisor from N
        self.current_N -= action

        # Check if the game is over after this move
        if self.current_N == 1 or self.is_prime(self.current_N):
            # Current player wins because opponent cannot make a valid move
            self.done = True
            return np.array([self.current_N], dtype=np.int32), 1, True, False, {}
        else:
            # Game continues; switch to the next player's turn
            return np.array([self.current_N], dtype=np.int32), -10, False, False, {}

    def render(self):
        # Return a string representation of the current game state
        return f"Current N: {self.current_N}"

    def valid_moves(self):
        # Return a list of valid proper divisors of the current N
        proper_divisors = [
            i for i in range(2, self.current_N) if self.current_N % i == 0
        ]
        return proper_divisors

    def is_prime(self, n):
        # Helper function to check if a number n is prime
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
