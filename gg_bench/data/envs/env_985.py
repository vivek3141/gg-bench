import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete actions from 1 to 100 (represented as 0 to 99)
        self.action_space = spaces.Discrete(100)

        # Observation space: current N (1 to 100)
        self.observation_space = spaces.Box(low=1, high=100, shape=(1,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Starting Number N between 50 and 100 inclusive
        self.N = random.randint(50, 100)
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.invalid_move = False
        return np.array([self.N], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, no further moves are allowed
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        chosen_divisor = action + 1  # Map action index to divisor (1 to 100)
        proper_divisors = self.get_proper_divisors(self.N)

        if chosen_divisor not in proper_divisors:
            # Invalid move
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Valid move
        self.N -= chosen_divisor

        # Check for win condition
        next_proper_divisors = self.get_proper_divisors(self.N)
        if len(next_proper_divisors) == 0:
            # Opponent cannot make a move; current player wins
            self.done = True
            reward = 1
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Continue the game
        reward = 0
        return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def render(self):
        # Return a visual representation of the game state as a string
        return f"Current Number (N): {self.N}\nCurrent Player: Player {self.current_player}"

    def valid_moves(self):
        # Return a list of valid action indices
        proper_divisors = self.get_proper_divisors(self.N)
        return [d - 1 for d in proper_divisors]  # Map divisors to action indices

    def get_proper_divisors(self, n):
        # Returns a list of proper divisors of n (excluding n itself)
        divisors = [i for i in range(1, n) if n % i == 0]
        return divisors
