import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: indices 0 to 48, corresponding to numbers 2 to 50
        self.action_space = spaces.Discrete(49)

        # Observation space: an array of length 49, values 0 or 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(49,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the number pool: array of ones, length 49
        self.number_pool = np.ones(49, dtype=np.int8)

        # Set current player: 1 or -1
        self.current_player = 1

        # Game over flag
        self.done = False

        # Return observation and info
        return self.number_pool.copy(), {}

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

    def step(self, action):
        if self.done:
            return self.number_pool.copy(), 0, True, False, {}

        # Check if action is valid
        if action < 0 or action >= 49:
            # Invalid action index
            self.done = True
            return self.number_pool.copy(), -10, True, False, {}

        if self.number_pool[action] != 1:
            # Number not in pool
            self.done = True
            return self.number_pool.copy(), -10, True, False, {}

        number = action + 2  # Map index to number

        # Check if number is prime
        if not self.is_prime(number):
            self.done = True
            return self.number_pool.copy(), -10, True, False, {}

        # Valid move, remove the selected number and its multiples from the number pool
        for i in range(49):
            number_i = i + 2
            if number_i % number == 0:
                self.number_pool[i] = 0

        # Check whether the opponent can make a move
        valid_moves_for_opponent = [
            i for i in range(49) if self.number_pool[i] == 1 and self.is_prime(i + 2)
        ]
        if not valid_moves_for_opponent:
            # Opponent cannot move, current player wins
            self.done = True
            return self.number_pool.copy(), 1, True, False, {}
        else:
            # Opponent can move, switch to opponent
            self.current_player *= -1
            return self.number_pool.copy(), 0, False, False, {}

    def render(self):
        available_numbers = [str(i + 2) for i in range(49) if self.number_pool[i] == 1]
        return "Available numbers: " + ", ".join(available_numbers)

    def valid_moves(self):
        return [
            i for i in range(49) if self.number_pool[i] == 1 and self.is_prime(i + 2)
        ]
