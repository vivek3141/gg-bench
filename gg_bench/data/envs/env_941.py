import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Numbers from 2 to 50 inclusive
        self.numbers = np.arange(2, 51)
        self.n_numbers = len(self.numbers)
        # Action space: integers from 0 to 48, representing indices of self.numbers
        self.action_space = spaces.Discrete(self.n_numbers)
        # Observation space: binary array indicating availability of numbers
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_numbers,), dtype=np.int8
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(self.n_numbers, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.available_numbers.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return self.available_numbers.copy(), 0, True, False, {}
        number_selected = self.numbers[action]
        idx = action
        if self.available_numbers[idx] == 0:
            # Invalid move
            self.done = True
            return self.available_numbers.copy(), -10, True, False, {}
        else:
            # Valid move
            self.remove_numbers(number_selected)
            # Check if opponent has valid moves
            opponent_valid_moves = self.valid_moves()
            if not opponent_valid_moves:
                # Opponent has no valid moves; current player wins
                self.done = True
                return self.available_numbers.copy(), 1, True, False, {}
            else:
                # Switch player
                self.current_player *= -1
                return self.available_numbers.copy(), 0, False, False, {}

    def remove_numbers(self, number):
        # Remove selected number, its factors, and multiples
        factors_multiples = self.get_factors_multiples(number)
        indices_to_remove = [
            np.where(self.numbers == n)[0][0]
            for n in factors_multiples
            if n in self.numbers
        ]
        self.available_numbers[indices_to_remove] = 0

    def get_factors_multiples(self, number):
        # Factors: numbers less than 'number' that divide 'number' evenly
        factors = [n for n in self.numbers if n < number and number % n == 0]
        # Multiples: numbers greater than 'number' that 'number' divides evenly into
        multiples = [n for n in self.numbers if n > number and n % number == 0]
        return factors + [number] + multiples

    def valid_moves(self):
        return [i for i in range(self.n_numbers) if self.available_numbers[i] == 1]

    def render(self):
        available = self.numbers[self.available_numbers == 1]
        available_list = available.tolist()
        return f"Available Numbers: {available_list}"
