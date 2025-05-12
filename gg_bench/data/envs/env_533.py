import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Starting number N
        self.N_start = 100
        self.N = self.N_start

        # List of primes up to N_start
        self.primes = self.generate_primes_up_to_n(self.N_start)
        self.n_actions = len(self.primes)

        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=np.array([1], dtype=np.int32),
            high=np.array([self.N_start], dtype=np.int32),
            shape=(1,),
            dtype=np.int32,
        )

        self.current_player = 1  # Can be 1 or -1
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_start
        self.current_player = 1
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Observation, info

    def step(self, action):
        if self.done:
            return (
                np.array([self.N], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        valid_actions = self.valid_moves()

        if len(valid_actions) == 0:
            # Current player cannot make a valid move
            self.done = True
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        if action not in valid_actions:
            # Invalid action
            self.done = True
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        prime = self.primes[action]

        # Valid move
        self.N = self.N // prime

        if self.N == 1:
            # Current player wins
            self.done = True
            return np.array([self.N], dtype=np.int32), 1, True, False, {}
        else:
            # Switch current player
            self.current_player *= -1
            return np.array([self.N], dtype=np.int32), -10, False, False, {}

    def render(self):
        return f"Current N: {self.N}, Current Player: {self.current_player}"

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [i for i, p in enumerate(self.primes) if self.N % p == 0]

    @staticmethod
    def generate_primes_up_to_n(n):
        # Sieve of Eratosthenes to generate primes up to n
        sieve = [True] * (n + 1)
        sieve[0:2] = [False, False]  # Zero and one are not primes
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        primes = [p for p, is_prime in enumerate(sieve) if is_prime]
        return primes
