import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, max_N=100):
        super(CustomEnv, self).__init__()
        self.max_N = max_N

        # Action space: choosing a divisor from 1 to max_N
        # Action indices 0 to max_N -1 correspond to divisor values 1 to max_N respectively
        self.action_space = spaces.Discrete(self.max_N)
        self.observation_space = spaces.Box(
            low=1, high=self.max_N, shape=(1,), dtype=np.int32
        )

        self.N = None
        self.current_player = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Set initial N
        if options and "N" in options:
            self.N = options["N"]
            if self.N < 2 or self.N > self.max_N:
                raise ValueError(f"N must be between 2 and {self.max_N}")
        else:
            self.N = np.random.randint(2, self.max_N + 1)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}
        divisor = action + 1  # Map action index to divisor

        # Check if divisor is a valid proper divisor of N
        if divisor >= self.N or self.N % divisor != 0:
            # Invalid move, player loses
            self.done = True
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        # Valid move
        self.N -= divisor

        if self.N == 0:
            # Current player wins
            self.done = True
            return np.array([self.N], dtype=np.int32), 1, True, False, {}

        # Check if N has any proper divisors
        if not self.get_proper_divisors(self.N):
            # Next player cannot move, current player wins
            self.done = True
            return np.array([self.N], dtype=np.int32), 1, True, False, {}

        # Switch player and continue
        self.current_player *= -1
        return np.array([self.N], dtype=np.int32), -10, False, False, {}

    def render(self):
        output = f"Current N: {self.N}\n"
        output += f"Player {1 if self.current_player ==1 else 2}'s turn.\n"
        proper_divisors = self.get_proper_divisors(self.N)
        if proper_divisors:
            output += f"Proper divisors of {self.N}: {proper_divisors}\n"
        else:
            output += f"No proper divisors for {self.N}.\n"
        return output

    def valid_moves(self):
        if self.done:
            return []
        proper_divisors = self.get_proper_divisors(self.N)
        # Map divisors to action indices
        return [d - 1 for d in proper_divisors]

    def get_proper_divisors(self, N):
        return [d for d in range(1, N) if N % d == 0]
