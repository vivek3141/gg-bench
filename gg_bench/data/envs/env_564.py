import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 49 possible numbers to choose from (2 to 50 inclusive)
        self.action_space = spaces.Discrete(49)
        self.observation_space = spaces.Box(low=0, high=1, shape=(49,), dtype=np.int8)

        # Initialize game state variables
        self.available_numbers = None
        self.current_player = None
        self.done = False

        # Initialize the environment
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the list of available numbers (1 means available, 0 means taken)
        self.available_numbers = np.ones(49, dtype=np.int8)

        # Set the starting player (1 or -1)
        self.current_player = 1

        self.done = False

        return self.available_numbers.copy(), {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self.available_numbers.copy(), 0, True, False, {}

        # Validate the action
        if action < 0 or action >= 49 or self.available_numbers[action] == 0:
            # Invalid move: action out of bounds or number not available
            self.done = True
            reward = -10  # Penalty for invalid move
            return self.available_numbers.copy(), reward, True, False, {}

        # Valid move
        selected_number = action + 2  # Map action index to the actual number (2 to 50)

        # Remove the selected number from available numbers
        self.available_numbers[action] = 0

        # Remove proper divisors of the selected number
        for idx in range(action):
            number = idx + 2
            if selected_number % number == 0 and self.available_numbers[idx] == 1:
                self.available_numbers[idx] = 0

        # Remove proper multiples of the selected number
        for idx in range(action + 1, 49):
            number = idx + 2
            if number % selected_number == 0 and self.available_numbers[idx] == 1:
                self.available_numbers[idx] = 0

        # Switch to the next player
        self.current_player *= -1

        # Check if the opponent has any valid moves
        if not any(self.available_numbers):
            # No valid moves left for the opponent; current player wins
            self.done = True
            reward = 1  # Reward for winning
            return self.available_numbers.copy(), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            return self.available_numbers.copy(), reward, False, False, {}

    def render(self):
        # Generate a string representation of the available numbers
        available_numbers = [
            str(idx + 2) for idx in range(49) if self.available_numbers[idx] == 1
        ]
        if available_numbers:
            return "Available Numbers:\n" + ", ".join(available_numbers)
        else:
            return "No available numbers left."

    def valid_moves(self):
        # Return a list of valid moves (indices of available numbers)
        return [idx for idx in range(49) if self.available_numbers[idx] == 1]
