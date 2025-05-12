import numpy as np
import gymnasium as gym
from gymnasium import spaces
from math import gcd


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers from 1 to 15, represented by indices 0 to 14
        self.action_space = spaces.Discrete(15)

        # Define observation space: a binary vector indicating available numbers
        self.observation_space = spaces.Box(low=0, high=1, shape=(15,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start with all numbers from 1 to 15 available
        self.numbers = np.ones(
            15, dtype=np.int32
        )  # 1 indicates the number is available
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.numbers.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return self.numbers.copy(), -10, True, False, {}

        # Check if action is valid
        if action < 0 or action >= 15 or self.numbers[action] == 0:
            # Invalid action: out of bounds or number not available
            self.done = True
            return self.numbers.copy(), -10, True, False, {}

        # Valid move
        selected_number = action + 1  # Convert index to number (1 to 15)

        # Remove selected number
        self.numbers[action] = 0

        # Remove numbers that share a common factor greater than 1 with the selected number
        for i in range(15):
            if self.numbers[i] == 1 and i != action:
                other_number = i + 1  # Convert index to number
                if gcd(selected_number, other_number) > 1:
                    self.numbers[i] = 0  # Remove the number

        # Check if the game is over (no numbers left)
        if np.sum(self.numbers) == 0:
            # Current player wins
            self.done = True
            reward = 1  # Reward for winning
            return self.numbers.copy(), reward, True, False, {}

        # Switch to the other player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        return (
            self.numbers.copy(),
            0,
            False,
            False,
            {},
        )  # Return observation, reward, done, truncated, info

    def render(self):
        # Return a string representation of the current game state
        available_numbers = [str(i + 1) for i in range(15) if self.numbers[i] == 1]
        display = "Available numbers: " + ", ".join(available_numbers)
        return display

    def valid_moves(self):
        # Return a list of indices corresponding to available numbers
        return [i for i in range(15) if self.numbers[i] == 1]
