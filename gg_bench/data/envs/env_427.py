import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=10):
        super(CustomEnv, self).__init__()
        self.N = N  # Maximum number in the initial list
        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.N,), dtype=np.int8
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            self.N, dtype=np.int8
        )  # Numbers 1 to N are available
        self.current_player = 1  # 1 or -1 to indicate players
        self.done = False
        return self.available_numbers.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.available_numbers.copy(), 0, True, False, {}
        if self.available_numbers[action] != 1:
            # Invalid move: number not available
            self.done = True
            return self.available_numbers.copy(), -10, True, False, {}
        # Valid move
        selected_number = action + 1  # Numbers are from 1 to N
        self.available_numbers[action] = 0  # Remove selected number

        # Remove factors and multiples
        numbers = np.arange(1, self.N + 1)
        indices_to_remove = np.where(
            ((selected_number % numbers == 0) | (numbers % selected_number == 0))
            & (self.available_numbers == 1)
        )[0]
        self.available_numbers[indices_to_remove] = 0

        # Check if the opponent has valid moves
        if not np.any(self.available_numbers == 1):
            # Current player wins
            self.done = True
            return self.available_numbers.copy(), 1, True, False, {}
        else:
            # Switch to the next player
            self.current_player *= -1
            return self.available_numbers.copy(), 0, False, False, {}

    def render(self):
        available_numbers = np.where(self.available_numbers == 1)[0] + 1
        return f"Available Numbers: {' '.join(map(str, available_numbers))}"

    def valid_moves(self):
        return np.where(self.available_numbers == 1)[0].tolist()
