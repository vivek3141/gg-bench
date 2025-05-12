import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N_start=60, N_max=1000):
        super(CustomEnv, self).__init__()

        # Starting number N
        self.N_start = N_start
        self.N_max = N_max
        self.N = N_start

        # Current player: 1 or 2 (for internal management)
        self.current_player = 1

        # Define action space: Actions correspond to possible divisors D = action + 2
        self.action_space = spaces.Discrete(self.N_max - 1)

        # Define observation space: Observation is the current number N
        self.observation_space = spaces.Box(
            low=1, high=self.N_max, shape=(1,), dtype=np.int32
        )

        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_start
        self.current_player = 1
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        D = action + 2  # action corresponds to divisor D = action + 2

        # Invalid move: D does not divide N or D <= 1 or D >= N
        if D <= 1 or D >= self.N or self.N % D != 0:
            reward = -10
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Valid move: update N
        self.N = self.N // D

        # Check for winning conditions
        if self.N == 1:
            # Current player wins by reducing N to 1
            reward = 1
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}
        elif self.is_prime(self.N):
            # Opponent cannot make a move; current player wins
            reward = 1
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Continue the game: switch current player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        reward = 0
        return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def render(self):
        return f"Current Number: {self.N}"

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for D in range(2, self.N):
            if self.N % D == 0:
                action = D - 2
                valid_actions.append(action)
        return valid_actions

    @staticmethod
    def is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
