import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, initial_n=20):
        super(CustomEnv, self).__init__()

        self.initial_n = initial_n
        self.current_n = self.initial_n

        # Generate all prime numbers up to initial_n
        self.prime_numbers = self.generate_primes(self.initial_n)

        # Action space: indices of primes in self.prime_numbers
        self.action_space = spaces.Discrete(len(self.prime_numbers))

        # Observation space: current N as an integer in [0, initial_n]
        self.observation_space = spaces.Box(
            low=np.array([0]),
            high=np.array([self.initial_n]),
            shape=(1,),
            dtype=np.int32,
        )

        self.current_player = 0  # 0 or 1, for internal management
        self.done = False

    def generate_primes(self, n):
        primes = []
        for num in range(2, n + 1):
            is_prime = True
            for i in range(2, int(np.sqrt(num)) + 1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
        return primes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_n = self.initial_n
        self.done = False
        self.current_player = 0  # Start with player 1
        return np.array([self.current_n], dtype=np.int32), {}  # observation, info

    def step(self, action):
        if self.done:
            return np.array([self.current_n], dtype=np.int32), 0, True, False, {}

        info = {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            self.done = True
            return np.array([self.current_n], dtype=np.int32), reward, True, False, info

        # Valid move
        prime = self.prime_numbers[action]
        self.current_n -= prime

        if self.current_n == 0:
            # Current player wins
            reward = 1
            self.done = True
            return np.array([self.current_n], dtype=np.int32), reward, True, False, info

        if self.current_n < 0:
            # Should not occur due to valid_moves check, but included for safety
            reward = -10
            self.done = True
            return np.array([self.current_n], dtype=np.int32), reward, True, False, info

        # Check if next player has any valid moves
        next_valid_moves = self.valid_primes()
        if not next_valid_moves:
            # Next player cannot move; current player wins
            reward = 1
            self.done = True
            return np.array([self.current_n], dtype=np.int32), reward, True, False, info

        # Switch to next player
        self.current_player = 1 - self.current_player
        reward = 0
        return np.array([self.current_n], dtype=np.int32), reward, False, False, info

    def valid_primes(self):
        return [p for p in self.prime_numbers if p <= self.current_n]

    def valid_moves(self):
        # Action indices corresponding to valid primes
        valid_primes = self.valid_primes()
        valid_actions = [self.prime_numbers.index(p) for p in valid_primes]
        return valid_actions

    def render(self):
        return f"Current number (N): {self.current_n}, Current player: Player {self.current_player + 1}"
