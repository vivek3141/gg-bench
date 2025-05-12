import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=60, max_number=1000):
        super(CustomEnv, self).__init__()

        self.max_number = max_number
        self.starting_number = starting_number
        self.current_number = starting_number
        self.done = False

        # Action space: actions correspond to selected factors from 0 to max_number
        # selected_factor = action
        # Valid actions are selected_factor >= 2
        self.action_space = spaces.Discrete(
            self.max_number + 1
        )  # Actions from 0 to max_number

        # Observation space is the current number
        self.observation_space = spaces.Box(
            low=2, high=self.max_number, shape=(1,), dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        selected_factor = action
        if self.done:
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        # Check if action is valid
        if selected_factor <= 1 or selected_factor > self.max_number:
            # Invalid action: selected factor is out of bounds
            self.done = True
            return np.array([self.current_number], dtype=np.int32), -10, True, False, {}
        if self.current_number % selected_factor != 0:
            # Invalid action: selected factor does not divide current number
            self.done = True
            return np.array([self.current_number], dtype=np.int32), -10, True, False, {}
        other_factor = self.current_number // selected_factor
        if other_factor <= 1:
            # Invalid action: other factor is less than or equal to 1
            self.done = True
            return np.array([self.current_number], dtype=np.int32), -10, True, False, {}

        # Valid move, update current number
        self.current_number = selected_factor

        # Check if current number is prime
        if self.is_prime(self.current_number):
            # Next player cannot move; current player wins
            self.done = True
            return np.array([self.current_number], dtype=np.int32), 1, True, False, {}
        else:
            # Game continues
            return np.array([self.current_number], dtype=np.int32), 0, False, False, {}

    def render(self):
        return f"Current number: {self.current_number}"

    def valid_moves(self):
        moves = []
        for selected_factor in range(2, self.max_number + 1):
            if self.current_number % selected_factor == 0:
                other_factor = self.current_number // selected_factor
                if other_factor > 1:
                    moves.append(selected_factor)
        return moves

    def is_prime(self, n):
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
