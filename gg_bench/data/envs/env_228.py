import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the initial number N
        self.initial_N = 100
        self.max_N = self.initial_N

        # Generate all prime numbers up to max_N
        self.primes_list = self._generate_primes_up_to_N(self.max_N)
        self.num_primes = len(self.primes_list)

        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_primes)
        # Observation space consists of [current N, current player]
        self.observation_space = spaces.Box(
            low=np.array([1, -1]), high=np.array([self.max_N, 1]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.initial_N
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N, self.current_player]), {}

    def step(self, action):
        if self.done:
            return np.array([self.N, self.current_player]), 0, True, False, {}

        prime = self.primes_list[action]

        if self.N % prime != 0:
            # Invalid move
            self.done = True
            reward = -10
            return np.array([self.N, self.current_player]), reward, True, False, {}

        # Valid move
        self.N = self.N // prime

        if self.N == 1:
            # Current player wins
            self.done = True
            reward = 1
            return np.array([self.N, self.current_player]), reward, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        reward = 0
        return np.array([self.N, self.current_player]), reward, False, False, {}

    def render(self):
        return f"Current N: {self.N}, Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        return [i for i, p in enumerate(self.primes_list) if self.N % p == 0]

    def _generate_primes_up_to_N(self, N):
        sieve = [True] * (N + 1)
        sieve[0:2] = [False, False]
        for i in range(2, int(N**0.5) + 1):
            if sieve[i]:
                sieve[i * i : N + 1 : i] = [False] * len(range(i * i, N + 1, i))
        return [x for x, is_prime in enumerate(sieve) if is_prime]
