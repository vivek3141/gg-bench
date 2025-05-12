import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Starting number N
        self.initial_N = 30
        self.N = self.initial_N
        self.current_player = 1  # Player 1 or Player 2

        # Generate list of prime numbers up to initial_N
        self.max_prime = self.initial_N
        self.primes = self._generate_primes(self.max_prime)

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.primes))
        self.observation_space = spaces.Box(
            low=np.array([1, -1], dtype=np.int32),
            high=np.array([self.initial_N, 1], dtype=np.int32),
            dtype=np.int32,
        )

    def _generate_primes(self, n):
        """Generate a list of prime numbers up to n inclusive."""
        primes = []
        for num in range(2, n + 1):
            if self._is_prime(num):
                primes.append(num)
        return primes

    def _is_prime(self, num):
        """Check if a number is prime."""
        if num < 2:
            return False
        for i in range(2, int(np.sqrt(num)) + 1):
            if num % i == 0:
                return False
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.initial_N
        self.current_player = 1
        observation = np.array([self.N, self.current_player], dtype=np.int32)
        return observation, {}

    def step(self, action):
        # Check if action is valid
        if action < 0 or action >= len(self.primes):
            reward = -10
            terminated = True
            truncated = False
            observation = np.array([self.N, self.current_player], dtype=np.int32)
            return observation, reward, terminated, truncated, {}

        prime = self.primes[action]

        # Check if prime divides N
        if self.N % prime != 0:
            # Invalid move
            reward = -10
            terminated = True
            truncated = False
            observation = np.array([self.N, self.current_player], dtype=np.int32)
            return observation, reward, terminated, truncated, {}

        # Valid move
        self.N = self.N // prime

        if self.N == 1:
            # Current player wins
            reward = 1
            terminated = True
            truncated = False
            observation = np.array([self.N, self.current_player], dtype=np.int32)
            return observation, reward, terminated, truncated, {}

        # Game continues
        reward = 0
        terminated = False
        truncated = False
        # Switch player
        self.current_player *= -1
        observation = np.array([self.N, self.current_player], dtype=np.int32)
        return observation, reward, terminated, truncated, {}

    def render(self):
        state_str = f"Current N: {self.N}\n"
        player_str = f"Player {1 if self.current_player == 1 else 2}'s turn"
        return state_str + player_str

    def valid_moves(self):
        valid_actions = []
        for idx, prime in enumerate(self.primes):
            if self.N % prime == 0:
                valid_actions.append(idx)
        return valid_actions
