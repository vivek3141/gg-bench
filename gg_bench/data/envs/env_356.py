import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Generate a list of prime numbers up to a certain limit
        self.primes_list = self.generate_primes(100)  # Primes up to 100

        # Define action space: actions correspond to indices of primes in primes_list
        self.action_space = spaces.Discrete(len(self.primes_list))

        # Define observation space: the current number N
        self.N_INITIAL = 60  # Starting number N
        self.N_MAX = self.N_INITIAL  # Maximum possible value of N
        self.observation_space = spaces.Box(
            low=1, high=self.N_MAX, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_INITIAL  # Reset the Current Number N to initial value
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game is not over
        observation = np.array([self.N], dtype=np.int32)
        return observation, {}  # Return initial observation and info

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        prime = self.primes_list[action]

        # Valid move: update N
        self.N = self.N // prime

        if self.N == 1:
            # Current player wins
            reward = 1
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}
        else:
            # Check if next player has any valid moves
            next_valid_moves = [
                i
                for i, p in enumerate(self.primes_list)
                if self.N % p == 0 and p <= self.N
            ]
            if not next_valid_moves:
                # Next player cannot move, current player wins
                reward = 1
                self.done = True
                return np.array([self.N], dtype=np.int32), reward, True, False, {}
            else:
                # Switch to next player
                self.current_player = 3 - self.current_player  # Switch between 1 and 2
                reward = -10  # Penalize valid move to incentivize shorter games
                return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def render(self):
        return (
            f"Current Player: Player {self.current_player}\nCurrent Number: {self.N}\n"
        )

    def valid_moves(self):
        # Return indices of primes that are valid moves (prime factors of N)
        return [
            i
            for i, prime in enumerate(self.primes_list)
            if self.N % prime == 0 and prime <= self.N
        ]

    @staticmethod
    def generate_primes(n):
        """Generate a list of prime numbers up to n."""
        sieve = [True] * (n + 1)
        for p in range(2, int(n**0.5) + 1):
            if sieve[p]:
                for i in range(p * p, n + 1, p):
                    sieve[i] = False
        return [p for p in range(2, n + 1) if sieve[p]]
