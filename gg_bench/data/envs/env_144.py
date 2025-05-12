import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(8), representing numbers 2 to 9
        self.action_space = spaces.Discrete(8)
        # Observation space: An array of size 9
        # Index 0: Cumulative product (1 to 999)
        # Indices 1-8: Availability of numbers 2 to 9 (1 if available, 0 if used)
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(9,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize cumulative product and available numbers
        self.cumulative_product = 1
        self.available_numbers = [2, 3, 4, 5, 6, 7, 8, 9]
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Initialize observation
        self.observation = np.zeros(9, dtype=np.int32)
        self.observation[0] = self.cumulative_product
        self.observation[1:] = 1  # All numbers are initially available

        return self.observation, {}

    def step(self, action):
        if self.done:
            return self.observation, 0, True, False, {}

        # Map action to selected number (action 0-7 maps to number 2-9)
        selected_number = action + 2

        # Check if the selected number is available
        if selected_number not in self.available_numbers:
            # Invalid move
            self.done = True
            return self.observation, -10, True, False, {}

        # Update the cumulative product
        self.cumulative_product *= selected_number

        # Remove the selected number from available numbers
        self.available_numbers.remove(selected_number)

        # Update observation
        self.observation[0] = self.cumulative_product
        self.observation[selected_number - 1] = 0  # Mark number as used

        # Check for losing condition
        if self.cumulative_product >= 1000:
            # Current player loses
            self.done = True
            return self.observation, -10, True, False, {}

        # Check if all numbers have been used
        if not self.available_numbers:
            # Game ends in a draw (unlikely in this game)
            self.done = True
            return self.observation, 0, True, False, {}

        # Switch player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        # Valid move; assign negative reward as per prompt
        return self.observation, -10, False, False, {}

    def render(self):
        state_str = f"Cumulative Product: {self.cumulative_product}\n"
        state_str += "Available Numbers: "
        state_str += " ".join(str(num) for num in self.available_numbers)
        return state_str

    def valid_moves(self):
        return [num - 2 for num in self.available_numbers]
