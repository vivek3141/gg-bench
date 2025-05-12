import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=50):
        super(CustomEnv, self).__init__()
        self.N = N
        # Actions correspond to selecting numbers from 2 to N
        # Action indices range from 0 to N - 2
        self.action_space = spaces.Discrete(self.N - 1)
        # Observation is an array indicating the availability of numbers from 2 to N
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.N - 1,), dtype=np.int8
        )
        self.available_numbers = None
        self.current_player = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers from 2 to N are initially available
        self.available_numbers = np.ones(self.N - 1, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.available_numbers.copy(), {}  # Observation and info

    def step(self, action):
        if self.done:
            # The game has already ended
            return self.available_numbers.copy(), 0, True, False, {}
        selected_number = action + 2  # Map action index to the actual number
        # Check if the selected number is available
        if self.available_numbers[action] == 0:
            # Invalid move: number not available
            self.done = True
            return self.available_numbers.copy(), -10, True, False, {}
        # Check if the selected number is prime
        if not self.is_prime(selected_number):
            # Invalid move: number is not prime
            self.done = True
            return self.available_numbers.copy(), -10, True, False, {}
        # Valid move: remove the selected prime and its multiples
        for multiple in range(selected_number, self.N + 1, selected_number):
            idx = multiple - 2
            if idx < self.N - 1:
                self.available_numbers[idx] = 0
        # Check if any prime numbers remain for the next player
        primes_left = any(
            self.available_numbers[idx] == 1 and self.is_prime(idx + 2)
            for idx in range(self.N - 1)
        )
        if not primes_left:
            # Current player wins
            self.done = True
            return self.available_numbers.copy(), 1, True, False, {}
        else:
            # Switch to the other player
            self.current_player = 1 if self.current_player == 2 else 2
            return self.available_numbers.copy(), 0, False, False, {}

    def render(self):
        # Visual representation of the current game state
        available_nums = [
            str(idx + 2)
            for idx in range(self.N - 1)
            if self.available_numbers[idx] == 1
        ]
        return "Available Numbers: [{}]".format(", ".join(available_nums))

    def valid_moves(self):
        # List of valid moves as indices in the action space
        return [
            idx
            for idx in range(self.N - 1)
            if self.available_numbers[idx] == 1 and self.is_prime(idx + 2)
        ]

    def is_prime(self, n):
        # Efficient check for primality
        if n <= 1:
            return False
        elif n <= 3:
            return True
        elif n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
