import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define the maximum number length N based on the recommended starting number (e.g., 123456 has 6 digits)
        self.N = 6  # Maximum number of digits
        # Define action space: positions in the number to delete (from 0 to N-1)
        self.action_space = spaces.Discrete(self.N)
        # Define observation space: the current number represented as digits, padded with -1 for unused positions
        self.observation_space = spaces.Box(
            low=-1, high=9, shape=(self.N,), dtype=np.int32
        )
        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Starting number (e.g., 123456)
        self.starting_number = 123456
        # Convert the starting number into a list of digits
        starting_digits = [int(d) for d in str(self.starting_number)]
        # Initialize current number digits with padding
        self.current_number_digits = starting_digits + [-1] * (
            self.N - len(starting_digits)
        )
        # Current player indicator (1 or -1)
        self.current_player = 1  # Player 1 starts
        self.done = False
        # Update current length
        self.current_length = len(starting_digits)
        return np.array(self.current_number_digits, dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return (
                np.array(self.current_number_digits, dtype=np.int32),
                -10,
                True,
                False,
                {},
            )
        # Get the actual digits (exclude padding)
        actual_digits = [d for d in self.current_number_digits if d != -1]
        self.current_length = len(actual_digits)
        # Check if the current number is single-digit at the start of the turn
        if self.current_length == 1:
            # Current player loses
            self.done = True
            return (
                np.array(self.current_number_digits, dtype=np.int32),
                -1,
                True,
                False,
                {},
            )
        # Check if the action is valid
        if action < 0 or action >= self.current_length:
            # Invalid action
            self.done = True
            return (
                np.array(self.current_number_digits, dtype=np.int32),
                -10,
                True,
                False,
                {},
            )
        # Attempt to delete the digit at position 'action'
        new_digits = actual_digits[:action] + actual_digits[action + 1 :]
        # Check for leading zeros
        if new_digits and new_digits[0] == 0:
            # Invalid move, results in leading zero
            self.done = True
            return (
                np.array(self.current_number_digits, dtype=np.int32),
                -10,
                True,
                False,
                {},
            )
        # Update current number digits with padding
        self.current_number_digits = new_digits + [-1] * (self.N - len(new_digits))
        # Check for win condition
        actual_digits = [d for d in self.current_number_digits if d != -1]
        if len(actual_digits) == 1 and 1 <= actual_digits[0] <= 9:
            # Current player wins
            self.done = True
            return (
                np.array(self.current_number_digits, dtype=np.int32),
                1,
                True,
                False,
                {},
            )
        # No win condition met, switch to the next player
        self.current_player *= -1
        return (
            np.array(self.current_number_digits, dtype=np.int32),
            -10,
            False,
            False,
            {},
        )

    def render(self):
        actual_digits = [str(d) for d in self.current_number_digits if d != -1]
        number_str = "".join(actual_digits)
        return f"Current number: {number_str}"

    def valid_moves(self):
        # Return a list of valid action indices
        actual_digits = [d for d in self.current_number_digits if d != -1]
        valid_actions = []
        for i in range(len(actual_digits)):
            # Create a potential new number by deleting digit at index i
            new_digits = actual_digits[:i] + actual_digits[i + 1 :]
            if new_digits:
                if new_digits[0] != 0:
                    valid_actions.append(i)
            else:
                # Deleting the last digit would leave an empty number, which is invalid
                pass
        return valid_actions
