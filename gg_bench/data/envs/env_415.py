import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Parameters
        self.N_MAX = 10000  # Maximum value for N
        self.starting_N = 60  # Default starting number N
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False

        # Generate list of prime numbers up to N_MAX
        self.prime_list = self._generate_primes(self.N_MAX)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.prime_list))
        self.observation_space = spaces.Box(
            low=1, high=self.N_MAX, shape=(1,), dtype=np.int64
        )

        # Initialize the game state
        self.reset()

    def _generate_primes(self, n):
        """Generate a list of prime numbers up to n (inclusive) using Sieve of Eratosthenes."""
        sieve = np.ones(n + 1, dtype=bool)
        sieve[0:2] = False  # Mark 0 and 1 as non-prime
        for i in range(2, int(np.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i * i : n + 1 : i] = False
        primes = np.flatnonzero(sieve)
        return primes.tolist()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.starting_N
        self.done = False
        self.current_player = 1  # Player 1 starts
        return np.array([self.N], dtype=np.int64), {}  # Observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return np.array([self.N], dtype=np.int64), 0, True, False, {}

        if action < 0 or action >= len(self.prime_list):
            # Invalid action index
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int64), reward, True, False, {}

        prime = self.prime_list[action]

        # Check if the selected prime is a valid divisor of N
        if self.N % prime != 0:
            # Invalid move
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int64), reward, True, False, {}

        # Valid move: Update N
        self.N = self.N // prime

        # Check for win condition
        if self.N == 1:
            # Current player wins by reducing N to 1
            self.done = True
            reward = 1
            return np.array([self.N], dtype=np.int64), reward, True, False, {}
        elif self.N in self.prime_list:
            # Next player cannot make a move; current player wins
            self.done = True
            reward = 1
            return np.array([self.N], dtype=np.int64), reward, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        reward = 0
        return np.array([self.N], dtype=np.int64), reward, False, False, {}

    def render(self):
        return (
            f"Current N: {self.N}, "
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = [
            idx for idx, prime in enumerate(self.prime_list) if self.N % prime == 0
        ]
        return valid_actions
