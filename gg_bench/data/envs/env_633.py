import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the prime numbers up to 100
        self.prime_numbers = [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
        ]

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.prime_numbers))
        self.N_max = 100
        self.observation_space = spaces.Box(
            low=1, high=self.N_max, shape=(1,), dtype=np.int32
        )

        # Initialize random number generator
        self.np_random = None
        self.seed()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Select a composite number between 30 and N_max
        composite_numbers = [
            i for i in range(30, self.N_max + 1) if self._is_composite(i)
        ]
        self.N = self.np_random.choice(composite_numbers)

        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def _is_composite(self, n):
        if n <= 3:
            return False
        if n % 2 == 0 or n % 3 == 0:
            return True
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return True
            i += 6
        return False

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        chosen_prime = self.prime_numbers[action]
        valid_primes = [p for p in self.prime_numbers if self.N % p == 0]

        if chosen_prime not in valid_primes:
            # Invalid move
            self.done = True
            return (
                np.array([self.N], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info
        else:
            self.N = self.N // chosen_prime
            if self.N == 1:
                # Current player wins
                self.done = True
                return (
                    np.array([self.N], dtype=np.int32),
                    1,
                    True,
                    False,
                    {},
                )
            else:
                # Switch to next player
                self.current_player = (
                    3 - self.current_player
                )  # Switches between 1 and 2
                return (
                    np.array([self.N], dtype=np.int32),
                    0,
                    False,
                    False,
                    {},
                )

    def render(self):
        return f"Current N: {self.N}, Current Player: {self.current_player}"

    def valid_moves(self):
        return [i for i, p in enumerate(self.prime_numbers) if self.N % p == 0]
