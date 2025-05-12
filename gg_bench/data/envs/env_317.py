import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the prime numbers to be used as possible actions
        self.primes = np.array(
            [
                2,
                3,
                5,
                7,
                11,
                13,
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
                83,
                89,
                97,
            ]
        )
        self.num_actions = len(self.primes)
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation is the current value of N
        # Assume N ranges from 1 to a maximum value
        self.max_N = 10000
        self.observation_space = spaces.Box(
            low=1, high=self.max_N, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Set starting N
        if options is not None and "starting_N" in options:
            self.N = options["starting_N"]
            if self.N < 2 or self.N > self.max_N:
                raise ValueError(f"starting_N must be between 2 and {self.max_N}")
        else:
            # Default starting N is 100
            self.N = 100

        self.current_player = 1  # Player 1 starts
        self.done = False

        return np.array([self.N], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        # Map action index to prime number
        prime = self.primes[action]

        # Check if move is valid
        if self.N % prime != 0:
            # Invalid move
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Perform the division
        self.N = self.N // prime

        # Check if N is reduced to 1
        if self.N == 1:
            self.done = True
            reward = 1
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Valid move, game continues
        reward = 0
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def render(self):
        return f"Current N: {self.N}, Player {self.current_player}'s turn."

    def valid_moves(self):
        # Return list of action indices corresponding to valid moves
        valid_moves = []
        for idx, prime in enumerate(self.primes):
            if self.N % prime == 0:
                valid_moves.append(idx)
        return valid_moves
