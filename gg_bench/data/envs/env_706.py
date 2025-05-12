import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Game parameters
        self.N_MAX = 1000  # Maximum value of N
        self.MIN_N = 10  # Minimum starting value of N
        self.MAX_N = 50  # Maximum starting value of N
        self.MAX_PRIME = 100  # Maximum prime number to consider

        # Generate list of prime numbers up to MAX_PRIME
        self.PRIMES = self._generate_primes(self.MAX_PRIME)
        self.num_primes = len(self.PRIMES)

        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_primes)
        self.observation_space = spaces.Box(
            low=1, high=self.N_MAX, shape=(1,), dtype=np.int32
        )

        # Initialize game state
        self.current_player = 1
        self.done = False
        self.N = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the shared number N
        self.N = np.array(
            [self.np_random.integers(self.MIN_N, self.MAX_N + 1)], dtype=np.int32
        )
        self.current_player = 1
        self.done = False
        return self.N.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.N.copy(), -10, True, False, {}  # Game is already over

        prime = self.PRIMES[action]
        if self.N[0] % prime != 0:
            # Invalid move: the chosen prime does not divide current N
            self.done = True
            return self.N.copy(), -10, True, False, {}  # Invalid move

        # Valid move: divide N by the chosen prime factor
        self.N[0] = self.N[0] // prime

        if self.N[0] == 1:
            # Current player wins
            self.done = True
            return self.N.copy(), 1, True, False, {}  # Player wins

        # Switch to the next player
        self.current_player = 3 - self.current_player  # Switch between player 1 and 2
        return self.N.copy(), 0, False, False, {}  # Continue game

    def render(self):
        return f"Current N: {self.N[0]}"

    def valid_moves(self):
        valid_actions = [
            idx for idx, prime in enumerate(self.PRIMES) if self.N[0] % prime == 0
        ]
        return valid_actions

    def _generate_primes(self, n):
        """Generate a list of prime numbers up to n."""
        sieve = [True] * (n + 1)
        for p in range(2, int(n**0.5) + 1):
            if sieve[p]:
                for i in range(p * p, n + 1, p):
                    sieve[i] = False
        primes = [p for p in range(2, n + 1) if sieve[p]]
        return primes
