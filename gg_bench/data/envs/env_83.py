import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers from 2 to 9 (indices 0 to 7)
        self.action_space = spaces.Discrete(8)  # Actions: indices 0-7
        self.action_to_number = [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ]  # Map action indices to numbers

        # Define observation space: current total
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([1000]), shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total = 1  # Starting total
        self.current_player = 1  # Player 1 starts
        return np.array([self.total], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        # Check if the opponent has already lost
        if self.is_prime(self.total):
            # Opponent made total prime, current player wins
            reward = 1  # Current player wins
            terminated = True
            truncated = False
            return (
                np.array([self.total], dtype=np.int32),
                reward,
                terminated,
                truncated,
                {},
            )

        # Validate the action
        if action not in range(8):
            reward = -10  # Penalty for invalid action
            terminated = True
            truncated = False
            return (
                np.array([self.total], dtype=np.int32),
                reward,
                terminated,
                truncated,
                {},
            )

        # Map action to number to add
        number_to_add = self.action_to_number[action]

        # Update the total
        self.total += number_to_add

        # Check if the current player loses
        if self.is_prime(self.total):
            reward = -10  # Penalty for losing
            terminated = True
            truncated = False
        else:
            reward = 0  # Valid move
            terminated = False
            truncated = False
            # Switch player
            self.current_player = 2 if self.current_player == 1 else 1

        return (
            np.array([self.total], dtype=np.int32),
            reward,
            terminated,
            truncated,
            {},
        )

    def render(self):
        return f"Current Total: {self.total}\nPlayer {self.current_player}'s turn.\n"

    def valid_moves(self):
        return list(range(8))  # All actions are always valid (indices 0-7)

    @staticmethod
    def is_prime(n):
        """Check if a number is prime."""
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
