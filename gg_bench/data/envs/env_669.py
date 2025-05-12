import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N_init=None):
        super(CustomEnv, self).__init__()

        # Generate list of prime numbers up to a certain limit
        self.prime_limit = 100  # Adjust the limit as needed
        self.primes_list = self.generate_primes(self.prime_limit)

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.primes_list))

        # Observation space: N is an integer between 1 and N_max
        self.N_max = 100  # Maximum possible value of N
        self.observation_space = spaces.Box(
            low=np.array([1], dtype=np.int32),
            high=np.array([self.N_max], dtype=np.int32),
            shape=(1,),
            dtype=np.int32,
        )

        self.N_init = N_init  # Allows setting an initial N if desired
        self.reset()

    def generate_primes(self, n):
        """Generate a list of prime numbers up to n using Sieve of Eratosthenes."""
        sieve = [True] * (n + 1)
        sieve[0:2] = [False, False]
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        primes = [i for i, prime in enumerate(sieve) if prime]
        return primes

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.N_init is not None:
            self.N = self.N_init
        else:
            # Choose a starting N between 30 and 100
            self.N = np.random.randint(30, 101)
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return np.array([self.N], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            reward = -10
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}
        else:
            prime = self.primes_list[action]
            self.N = self.N // prime
            if self.N == 1:
                # Current player wins
                reward = 1
                self.done = True
                return np.array([self.N], dtype=np.int32), reward, True, False, {}
            else:
                # Continue the game
                self.current_player *= -1  # Switch player
                return np.array([self.N], dtype=np.int32), 0, False, False, {}

    def render(self):
        return f"Current N: {self.N}, Current player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        """Return a list of valid action indices based on the current N."""
        factors = self.get_prime_factors(self.N)
        valid_actions = [self.primes_list.index(factor) for factor in factors]
        return valid_actions

    def get_prime_factors(self, n):
        """Return the unique prime factors of n."""
        factors = set()
        temp_n = n
        for p in self.primes_list:
            if p * p > temp_n:
                break
            if temp_n % p == 0:
                factors.add(p)
                while temp_n % p == 0:
                    temp_n = temp_n // p
        if temp_n > 1:
            # temp_n is a prime number greater than sqrt(n)
            factors.add(temp_n)
        return list(factors)
