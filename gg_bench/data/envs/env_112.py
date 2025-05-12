import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

MAX_N = 1000


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(MAX_N + 1)
        self.observation_space = spaces.Box(
            low=1, high=MAX_N, shape=(1,), dtype=np.int32
        )

        self.N = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "starting_n" in options:
            self.N = options["starting_n"]
            if self.N < 2 or self.N > MAX_N:
                raise ValueError(f"starting_n must be between 2 and {MAX_N}")
        else:
            self.N = random.randint(3, MAX_N)
        self.done = False
        return np.array([self.N], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # Should not call step after the game is over
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        if action < 0 or action > MAX_N:
            # Action is out of bounds
            reward = -10
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        divisor = action + 1  # Map action index to divisor
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Subtract the divisor from N
        self.N -= divisor

        # Check for game end conditions
        if self.N <= 1:
            # Opponent cannot make a move; current player wins
            reward = 1
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Check if the next player has valid moves
        next_valid_actions = self._get_valid_actions(self.N)
        if not next_valid_actions:
            # Opponent cannot make a move; current player wins
            reward = 1
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, True, False, {}
        else:
            # Valid move made; switch to the next player
            reward = -10
            self.done = False
            return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def render(self):
        return f"Current N: {self.N}"

    def valid_moves(self):
        return self._get_valid_actions(self.N)

    def _get_valid_actions(self, N):
        proper_divisors = [d for d in range(2, N) if N % d == 0]
        action_indices = [d - 1 for d in proper_divisors]
        return action_indices
