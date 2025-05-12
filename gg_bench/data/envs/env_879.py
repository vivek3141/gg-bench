import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_N=100):
        super(CustomEnv, self).__init__()

        self.N_max = starting_N  # Maximum N, adjustable if needed
        self.starting_N = starting_N

        # Define action and observation space
        self.action_space = spaces.Discrete(
            self.N_max + 1
        )  # Actions from 0 to N_max inclusive
        self.observation_space = spaces.Box(
            low=1, high=self.N_max, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.N = self.starting_N
        self.current_player = 1  # Player 1 starts
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.starting_N
        self.current_player = 1
        self.done = False
        return (
            np.array([self.N], dtype=np.int32),
            {},
        )  # Return initial observation and info

    def step(self, action):
        if self.done:
            return (
                np.array([self.N], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        valid_moves = self.valid_moves()

        if action not in valid_moves:
            # Invalid move
            self.done = True
            reward = -10
            return (
                np.array([self.N], dtype=np.int32),
                reward,
                True,
                False,
                {},  # Terminated, not truncated
            )
        else:
            # Valid move
            self.N = self.N // action
            if self.N == 1:
                # Current player wins by reducing N to 1
                self.done = True
                reward = 1
                return (
                    np.array([self.N], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )
            elif self.is_prime(self.N):
                # Opponent cannot make a move; current player wins
                self.done = True
                reward = 1
                return (
                    np.array([self.N], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )
            else:
                # Continue the game
                reward = -10
                self.current_player *= -1  # Switch player
                return (
                    np.array([self.N], dtype=np.int32),
                    reward,
                    False,
                    False,
                    {},
                )

    def render(self):
        return f"Current N: {self.N}, Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        # Return a list of valid actions (proper divisors of N)
        divisors = self.get_proper_divisors(self.N)
        return divisors

    def get_proper_divisors(self, n):
        # Compute proper divisors of n (excluding 1 and n)
        if n <= 3:
            return []
        divisors = [i for i in range(2, n) if n % i == 0]
        return divisors

    def is_prime(self, n):
        # Check if n is a prime number
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
