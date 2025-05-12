import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sympy import primerange


class CustomEnv(gym.Env):
    def __init__(self, N_init=30):
        super(CustomEnv, self).__init__()

        # Initialize the starting number N
        self.N_init = N_init
        self.N = self.N_init

        # Generate list of prime numbers up to N_init
        # This will be our action space mapping
        max_prime = self.N_init
        self.primes_list = list(primerange(2, max_prime + 1))
        self.prime_indices = {prime: idx for idx, prime in enumerate(self.primes_list)}

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.primes_list))
        self.observation_space = spaces.Box(
            low=1, high=self.N_init, shape=(), dtype=np.int32
        )

        # Initialize variables
        self.current_player = 1  # Player 1 starts
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_init
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.N, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.N, 0, True, False, {}

        # Map action index to prime number
        prime = self.primes_list[action]

        # Check if prime is a valid prime factor of N
        if self.N % prime != 0:
            self.done = True
            return (
                self.N,
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move, update N
        self.N = self.N // prime

        # Check for win condition
        if self.N == 1:
            self.done = True
            return self.N, 1, True, False, {}

        # Game continues, switch player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        return self.N, 0, False, False, {}

    def render(self):
        return f"Current N: {self.N}, Next player: Player {self.current_player}"

    def valid_moves(self):
        valid_primes = [prime for prime in self.primes_list if self.N % prime == 0]
        valid_actions = [self.prime_indices[prime] for prime in valid_primes]
        return valid_actions
