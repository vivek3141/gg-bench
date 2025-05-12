import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.N_MIN = 4
        self.N_MAX = 1000

        # Define action and observation space
        # Action space: integers from 0 to N_MAX inclusive
        self.action_space = spaces.Discrete(self.N_MAX + 1)
        # Observation space: [N, current_player], where N is between N_MIN and N_MAX, current_player is -1 or 1
        self.observation_space = spaces.Box(
            low=np.array([1, -1]), high=np.array([self.N_MAX, 1]), dtype=np.float32
        )

        # Precompute list of primes up to N_MAX
        self.primes = self._generate_primes(self.N_MAX)
        self.is_prime_array = self._generate_is_prime_array(self.N_MAX)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = random.randint(self.N_MIN, self.N_MAX)
        self.current_player = 1
        self.done = False
        return np.array([self.N, self.current_player], dtype=np.float32), {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.N, self.current_player], dtype=np.float32),
                -10,
                True,
                False,
                {},
            )

        # Validate action
        if (
            action < 2
            or action > self.N
            or self.N % action != 0
            or not self.is_prime(action)
        ):
            self.done = True
            return (
                np.array([self.N, self.current_player], dtype=np.float32),
                -10,
                True,
                False,
                {},
            )

        # Valid action: update N
        self.N = self.N // action

        # Check if opponent has any valid moves
        opponent_valid_moves = self.get_prime_factors(self.N)
        if not opponent_valid_moves:
            # Current player wins
            reward = 1
            self.done = True
            return (
                np.array([self.N, self.current_player], dtype=np.float32),
                reward,
                True,
                False,
                {},
            )

        # Game continues
        self.current_player *= -1
        return (
            np.array([self.N, self.current_player], dtype=np.float32),
            0,
            False,
            False,
            {},
        )

    def render(self):
        return f"Current N: {self.N}\nCurrent Player: {self.current_player}"

    def valid_moves(self):
        return self.get_prime_factors(self.N)

    def is_prime(self, n):
        if n < 2 or n > self.N_MAX:
            return False
        return self.is_prime_array[n]

    def get_prime_factors(self, n):
        factors = []
        for p in self.primes:
            if p > n:
                break
            if n % p == 0:
                factors.append(p)
        return factors

    def _generate_primes(self, n):
        """Sieve of Eratosthenes"""
        sieve = [True] * (n + 1)
        sieve[0:2] = [False, False]
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i * i : n + 1 : i] = [False] * len(range(i * i, n + 1, i))
        primes = [i for i, prime in enumerate(sieve) if prime]
        return primes

    def _generate_is_prime_array(self, n):
        """Generate array where index represents the number and value is True if prime"""
        is_prime_array = [True] * (n + 1)
        is_prime_array[0:2] = [False, False]
        for i in range(2, int(n**0.5) + 1):
            if is_prime_array[i]:
                is_prime_array[i * i : n + 1 : i] = [False] * len(
                    range(i * i, n + 1, i)
                )
        return is_prime_array
