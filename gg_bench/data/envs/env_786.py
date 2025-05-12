import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: digits 1-9 represented as actions 0-8
        self.action_space = spaces.Discrete(9)

        # Observation space: Fixed-length array to represent the shared number
        # Maximum length of the shared number is set to 20 digits
        self.max_length = 20
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.max_length,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Shared number represented as a fixed-size array of digits
        self.shared_number = np.zeros(self.max_length, dtype=np.int32)
        self.current_index = 0  # Pointer to the next empty position in shared_number
        self.current_player = 1  # Player 1 starts first
        self.done = False
        return self.shared_number.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, no further moves can be made
            return self.shared_number.copy(), -10, True, False, {}

        # Map action to digit (1-9)
        if action < 0 or action > 8:
            # Invalid action
            return self.shared_number.copy(), -10, True, False, {}

        digit = action + 1

        # Check if we have space to append the digit
        if self.current_index >= self.max_length:
            # Maximum length reached, current player loses
            self.done = True
            return self.shared_number.copy(), -10, True, False, {}

        # Append the digit to the shared number
        self.shared_number[self.current_index] = digit
        self.current_index += 1

        # Build the shared number as an integer
        # Convert the digits array up to current_index to a number
        current_number = int(
            "".join(map(str, self.shared_number[: self.current_index]))
        )

        # Check for divisibility by 7
        if current_number % 7 == 0:
            # Current player loses
            self.done = True
            reward = -10  # Losing penalty
            terminated = True
        else:
            # Game continues
            reward = -10  # Penalty for valid move
            terminated = False
            # Switch to the other player
            self.current_player *= -1

        truncated = False  # Game cannot be truncated
        return self.shared_number.copy(), reward, terminated, truncated, {}

    def render(self):
        # Construct the shared number as a string
        number_str = "".join(map(str, self.shared_number[: self.current_index]))
        if not number_str:
            number_str = "[empty]"
        return f"Shared Number: {number_str}"

    def valid_moves(self):
        # All digits 1-9 are always available as actions 0-8
        if self.done or self.current_index >= self.max_length:
            # No valid moves if the game is over or max length reached
            return []
        else:
            return list(range(9))
