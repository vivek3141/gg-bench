import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, initial_number=123456789, divisor=3, max_num_digits=20):
        super(CustomEnv, self).__init__()

        # Maximum number of digits allowed in the game
        self.max_num_digits = max_num_digits

        # Define action space: Positions of digits to delete (0-indexed)
        self.action_space = spaces.Discrete(self.max_num_digits)

        # Define observation space: Array of digits, padded with zeros if necessary
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.max_num_digits,), dtype=np.int32
        )

        # Game parameters
        self.initial_number = initial_number
        self.divisor = divisor

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        # Reset the random seed if provided
        super().reset(seed=seed)

        # Set the current number to the initial number and convert to list of digits
        self.current_number = self.initial_number
        self.number_digits = self._number_to_digits(self.current_number)

        # Game status
        self.done = False

        # Return the initial observation and info dictionary
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            # If the game is over, no further action can be taken
            return self._get_observation(), 0, self.done, False, {}

        # Get the list of valid actions
        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move: The player loses the game
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, {}

        # Perform the digit deletion
        new_digits = self.number_digits[:action] + self.number_digits[action + 1 :]

        # Check if the new number is valid
        if not new_digits:
            # No digits left after deletion: Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, {}

        new_number = self._digits_to_number(new_digits)

        # Update the game state
        self.current_number = new_number
        self.number_digits = new_digits

        # Check if the next player has any valid moves
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            # Current player wins the game
            self.done = True
            reward = 1
            return self._get_observation(), reward, self.done, False, {}

        # Game continues: No reward
        reward = 0
        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        # Return a string representation of the current number
        return "".join(map(str, self.number_digits))

    def valid_moves(self):
        # Determine valid moves: Positions where deleting a digit results in a number divisible by the divisor
        valid_actions = []
        for idx in range(len(self.number_digits)):
            # Delete the digit at index idx
            new_digits = self.number_digits[:idx] + self.number_digits[idx + 1 :]

            if not new_digits:
                # Cannot have an empty number
                continue

            new_number = self._digits_to_number(new_digits)
            if new_number % self.divisor == 0:
                # Valid move found
                valid_actions.append(idx)
        return valid_actions

    def _get_observation(self):
        # Create an array of digits padded with zeros to match max_num_digits
        digits = self.number_digits
        padding_length = self.max_num_digits - len(digits)
        padded_digits = digits + [0] * padding_length
        observation = np.array(padded_digits, dtype=np.int32)
        return observation

    @staticmethod
    def _number_to_digits(number):
        # Convert a number to a list of its digits
        return [int(d) for d in str(number)]

    @staticmethod
    def _digits_to_number(digits):
        # Convert a list of digits back to a number
        return int("".join(map(str, digits)))
