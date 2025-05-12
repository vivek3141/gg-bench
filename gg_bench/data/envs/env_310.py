import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sympy


class CustomEnv(gym.Env):
    def __init__(self, N=None):
        super(CustomEnv, self).__init__()

        # Configuration
        self.N_MAX = 1000
        self.MAX_PRIME = 100  # Considering primes less than 100

        # Generate list of prime numbers up to MAX_PRIME
        self.primes_list = list(sympy.primerange(2, self.MAX_PRIME))
        self.prime_to_action = {p: i for i, p in enumerate(self.primes_list)}
        self.action_to_prime = {i: p for i, p in enumerate(self.primes_list)}

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.primes_list))
        self.observation_space = spaces.Box(
            low=1, high=self.N_MAX, shape=(1,), dtype=np.int64
        )

        # Set initial N
        self.initial_N = N if N is not None else 60  # Default starting N
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "N" in options:
            self.N = options["N"]
        else:
            self.N = self.initial_N
        self.current_player = 1
        self.done = False
        return np.array([self.N], dtype=np.int64), {}  # observation, info

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int64), 0, True, False, {}
        if action < 0 or action >= len(self.primes_list):
            # Invalid action index
            reward = -10
            self.done = True
            return np.array([self.N], dtype=np.int64), reward, True, False, {}
        p = self.action_to_prime[action]
        if self.N % p != 0:
            # Invalid move: the chosen prime does not divide N
            reward = -10
            self.done = True
            return np.array([self.N], dtype=np.int64), reward, True, False, {}
        # Valid move
        self.N = self.N // p  # Integer division
        if self.N == 1:
            # Current player wins
            reward = 1
            self.done = True
            return np.array([self.N], dtype=np.int64), reward, True, False, {}
        else:
            # Continue game
            reward = 0
            self.current_player = 2 if self.current_player == 1 else 1
            return np.array([self.N], dtype=np.int64), reward, False, False, {}

    def render(self):
        return f"Current N: {self.N}, Player {self.current_player}'s turn"

    def valid_moves(self):
        factors = sympy.primefactors(self.N)
        valid_actions = [
            self.prime_to_action[p] for p in factors if p in self.prime_to_action
        ]
        return valid_actions
