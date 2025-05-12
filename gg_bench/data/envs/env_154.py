import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the list of the first N_PRIMES primes
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        self.N_PRIMES = len(self.primes)

        # Define action and observation space
        self.action_space = spaces.Discrete(self.N_PRIMES)
        self.MAX_SHARED_NUMBER = int(1e6)
        self.observation_space = spaces.Box(
            low=1, high=self.MAX_SHARED_NUMBER, shape=(1,), dtype=np.int64
        )

        # Initialize the game state
        self.shared_number = None
        self.current_player = 1
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1
        self.done = False

        # Generate a starting number with at least two prime factors
        self.shared_number = self._generate_starting_number()

        observation = np.array([self.shared_number], dtype=np.int64)
        return observation, {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.shared_number], dtype=np.int64),
                -10,
                True,
                False,
                {},
            )

        # Map action index to prime number
        if action < 0 or action >= self.N_PRIMES:
            reward = -10
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int64),
                reward,
                True,
                False,
                {},
            )

        prime = self.primes[action]

        if self.shared_number % prime != 0 or prime <= 1:
            reward = -10
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int64),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        self.shared_number = self.shared_number // prime

        if self.shared_number == 1:
            reward = 1
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int64),
                reward,
                True,
                False,
                {},
            )
        else:
            # Switch player
            self.current_player = 3 - self.current_player
            reward = 0
            return (
                np.array([self.shared_number], dtype=np.int64),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        render_str = f"Current shared number: {self.shared_number}\n"
        prime_factors = self._prime_factors(self.shared_number)
        render_str += f"Prime factors: {', '.join(map(str, prime_factors))}\n"
        render_str += f"Current player's turn: Player {self.current_player}\n"
        return render_str

    def valid_moves(self):
        factors = self._prime_factors(self.shared_number)
        valid_primes = set(factors) & set(self.primes)
        return [self.primes.index(p) for p in valid_primes]

    def _generate_starting_number(self):
        # Generate a starting number with at least two prime factors
        while True:
            # Randomly select two distinct primes
            prime_factors = np.random.choice(self.primes, size=2, replace=False)
            # Randomly decide how many times each prime appears
            exponents = np.random.randint(1, 4, size=2)
            number = np.prod(np.power(prime_factors, exponents))
            if number <= self.MAX_SHARED_NUMBER:
                return int(number)

    def _prime_factors(self, n):
        factors = []
        temp_n = n
        for prime in self.primes:
            while temp_n % prime == 0:
                factors.append(prime)
                temp_n = temp_n // prime
            if temp_n == 1:
                break
        if temp_n > 1:
            # Include remaining prime factors outside of predefined primes
            possible_prime = prime + 1
            while possible_prime * possible_prime <= temp_n:
                if temp_n % possible_prime == 0:
                    factors.append(possible_prime)
                    temp_n = temp_n // possible_prime
                else:
                    possible_prime += 1
            if temp_n > 1:
                factors.append(temp_n)
        return sorted(set(factors))
