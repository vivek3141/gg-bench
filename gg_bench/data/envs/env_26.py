import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from sympy import isprime


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: actions from 0 to 98, corresponding to divisors from 2 to 100
        self.action_space = spaces.Discrete(99)

        # Observation space: current shared number between 1 and 100
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([100]), shape=(1,), dtype=np.int32
        )

        # Initialize variables
        self.current_number = None
        self.current_player = 1  # Player 1 starts
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate a random composite number between 50 and 100 with at least one proper divisor
        composite_numbers = [n for n in range(50, 101) if not isprime(n)]
        while True:
            self.current_number = random.choice(composite_numbers)
            if len(self.get_proper_divisors(self.current_number)) > 0:
                break  # Found a suitable starting number

        self.current_player = 1  # Player 1 starts
        self.done = False

        return np.array([self.current_number], dtype=np.int32), {}  # observation, info

    def step(self, action):
        if self.done:
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        # Map action to divisor (divisor = action + 2)
        divisor = action + 2

        valid_divisors = self.get_proper_divisors(self.current_number)

        if divisor not in valid_divisors:
            # Invalid move
            self.done = True
            return np.array([self.current_number], dtype=np.int32), -10, True, False, {}

        # Valid move
        self.current_number = self.current_number // divisor

        if self.current_number == 1:
            # Current player wins
            self.done = True
            return np.array([self.current_number], dtype=np.int32), 1, True, False, {}

        # Check if the opponent has any valid moves
        opponent_valid_moves = self.get_proper_divisors(self.current_number)

        if len(opponent_valid_moves) == 0:
            # Opponent cannot move, current player wins
            self.done = True
            return np.array([self.current_number], dtype=np.int32), 1, True, False, {}

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        return np.array([self.current_number], dtype=np.int32), 0, False, False, {}

    def render(self):
        return f"Current number: {self.current_number}\nCurrent player: Player {self.current_player}"

    def valid_moves(self):
        # Return list of valid actions corresponding to proper divisors
        valid_divisors = self.get_proper_divisors(self.current_number)
        valid_actions = [
            d - 2 for d in valid_divisors
        ]  # Map divisors to action indices
        return valid_actions

    def get_proper_divisors(self, n):
        # Proper divisors are integers greater than 1 and less than n that divide n evenly
        return [i for i in range(2, n) if n % i == 0]
