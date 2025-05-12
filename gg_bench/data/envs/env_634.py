import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum N value
        self.max_N = 100  # Maximum value of N

        # Generate all primes up to max_N
        self.primes_list = self._generate_primes_up_to(self.max_N)
        self.num_actions = len(self.primes_list)
        self.action_space = spaces.Discrete(self.num_actions)

        self.observation_space = spaces.Box(
            low=1, high=self.max_N, shape=(1,), dtype=np.int32
        )

        # Initialize state
        self.N = None
        self.current_player = 1
        self.done = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Extract options
        if options is not None and "starting_N" in options:
            self.N = options["starting_N"]
            # Check that starting_N is valid
            if not (
                self.N > 1
                and self.N <= self.max_N
                and len(self._get_prime_factors(self.N)) >= 2
            ):
                raise ValueError(
                    f"starting_N must be a composite number greater than 1 and less than or equal to {self.max_N}, with at least two prime factors."
                )
        else:
            # Default starting N
            self.N = 30  # Default starting number

        self.current_player = 1  # Player 1 starts
        self.done = False

        # Return observation and info
        observation = np.array([self.N], dtype=np.int32)
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            raise Exception("Game is over. Please reset the environment.")
        if action < 0 or action >= self.num_actions:
            reward = -10
            self.done = True
            info = {}
            observation = np.array([self.N], dtype=np.int32)
            return observation, reward, self.done, False, info

        prime = self.primes_list[action]
        if self.N % prime != 0:
            # Invalid move
            reward = -10
            self.done = True
            info = {}
            observation = np.array([self.N], dtype=np.int32)
            return observation, reward, self.done, False, info
        else:
            # Valid move
            self.N = self.N // prime
            observation = np.array([self.N], dtype=np.int32)

            if self.N == 1:
                # Current player wins
                reward = 1
                self.done = True
                info = {}
                return observation, reward, self.done, False, info
            else:
                # Game continues
                reward = 0
                self.done = False
                # Switch player
                self.current_player = 2 if self.current_player == 1 else 1
                info = {}
                return observation, reward, self.done, False, info

    def render(self):
        # Return a visual representation of the environment state as a string
        output = f"Current N is {self.N}.\n"
        prime_factors = self._get_prime_factors(self.N)
        output += f"Prime factors are: {', '.join(map(str, prime_factors))}.\n"
        output += f"Player {self.current_player}'s turn.\n"
        return output

    def valid_moves(self):
        # Return a list of valid action indices
        prime_factors = self._get_prime_factors(self.N)
        valid_actions = []
        for i, prime in enumerate(self.primes_list):
            if prime in prime_factors:
                valid_actions.append(i)
        return valid_actions

    def _get_prime_factors(self, n):
        prime_factors = []
        temp_n = n

        for prime in self.primes_list:
            if prime > temp_n:
                break
            if temp_n % prime == 0:
                prime_factors.append(prime)
        return prime_factors

    def _generate_primes_up_to(self, n):
        sieve = [True] * (n + 1)
        sieve[0:2] = [False, False]
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i * i : n + 1 : i] = [False] * len(range(i * i, n + 1, i))
        primes = [i for i, is_prime in enumerate(sieve) if is_prime]
        return primes
