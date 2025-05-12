import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N_initial=30):
        super(CustomEnv, self).__init__()

        self.N_initial = N_initial
        self.N = None  # Current value of N
        self.used_factors = set()
        self.done = False

        # Define action space: actions correspond to potential factors from 2 to N_initial
        self.action_space = spaces.Discrete(
            self.N_initial + 1
        )  # Actions from 0 to N_initial inclusive

        # Define observation space: [current N (scaled), indicator vector for used factors]
        # Observation size is N_initial + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.N_initial + 1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_initial
        self.used_factors = set()
        self.done = False

        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        # Check for invalid action
        if not self._is_valid_action(action):
            self.done = True
            return self._get_observation(), -10.0, True, False, {}

        # Perform the action
        self.N = self.N // action
        self.used_factors.add(action)

        # Check for win condition
        if self.N == 1:
            self.done = True
            return self._get_observation(), 1.0, True, False, {}

        # Check if there are any valid moves left
        if not self.valid_moves():
            self.done = True
            return self._get_observation(), -1.0, True, False, {}

        # Game continues
        return self._get_observation(), -10.0, False, False, {}

    def render(self):
        state = f"Current N: {self.N}\n"
        state += f"Used Factors: {sorted(self.used_factors)}\n"
        return state

    def valid_moves(self):
        # Valid actions are factors of N (excluding 1 and N) that have not been used
        factors = self._get_factors(self.N)
        valid_actions = [a for a in factors if a not in self.used_factors]
        return valid_actions

    def _get_observation(self):
        # Observation is an array: [scaled N, used factors indicator vector]
        obs = np.zeros(self.N_initial + 1, dtype=np.float32)
        obs[0] = self.N / self.N_initial  # Scale N to [0,1]
        for factor in self.used_factors:
            if factor <= self.N_initial:
                obs[factor] = 1.0
        return obs

    def _is_valid_action(self, action):
        # Action must be an integer in the valid range
        if not isinstance(action, int) or action < 0 or action > self.N_initial:
            return False
        # Exclude 1 and N itself
        if action == 1 or action == self.N:
            return False
        # Must be a factor of N
        if self.N % action != 0:
            return False
        # Must not have been used before
        if action in self.used_factors:
            return False
        return True

    def _get_factors(self, n):
        # Get factors of n excluding 1 and n itself
        factors = set()
        for i in range(2, n):
            if n % i == 0:
                factors.add(i)
        return factors
