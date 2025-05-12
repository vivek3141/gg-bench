import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Choose to append '0' or '1'
        self.action_space = spaces.Discrete(2)
        # Observation space: Binary string of length 20 (-1 indicates unfilled positions)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.binary_string = np.full(
            20, -1, dtype=np.int8
        )  # Start with an empty binary string
        self.current_position = 0  # Next index to append a digit
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            self.binary_string.copy(),
            {},
        )  # Return initial observation and empty info

    def step(self, action):
        # Check for invalid actions
        if action not in [0, 1]:
            return self.binary_string.copy(), -10, True, False, {}  # Invalid move
        if self.done:
            return self.binary_string.copy(), -10, True, False, {}  # Game already over
        if self.current_position >= 20:
            self.done = True
            return (
                self.binary_string.copy(),
                -10,
                True,
                False,
                {},
            )  # Maximum length reached unexpectedly

        # Append the action to the binary string
        self.binary_string[self.current_position] = action
        self.current_position += 1

        # Convert the current binary string to a decimal number
        binary_list = self.binary_string[: self.current_position]
        binary_str = "".join(str(bit) for bit in binary_list)
        decimal_value = int(binary_str, 2)

        # Check for winning condition
        if decimal_value % 5 == 0:
            self.done = True
            return self.binary_string.copy(), 1, True, False, {}  # Current player wins

        # Check if maximum length reached without a winner
        if self.current_position >= 20:
            self.done = True
            return self.binary_string.copy(), 0, True, False, {}  # Game ends in a draw

        # Switch to the other player
        self.current_player *= -1  # Toggle between 1 and -1
        return (
            self.binary_string.copy(),
            -10,
            False,
            False,
            {},
        )  # Valid move, continue game

    def render(self):
        # Generate a visual representation of the current state
        binary_list = self.binary_string[: self.current_position]
        binary_str = "".join(str(bit) for bit in binary_list)
        decimal_value = int(binary_str, 2) if binary_str else 0
        return f"Current Binary String: {binary_str} (Decimal: {decimal_value})"

    def valid_moves(self):
        # Return a list of valid moves ([0, 1]) if the game is not over
        return [0, 1] if not self.done and self.current_position < 20 else []
