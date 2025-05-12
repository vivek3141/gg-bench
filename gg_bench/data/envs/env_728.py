import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_divisor=7):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Actions correspond to digits 1-9
        self.observation_space = spaces.Box(low=0, high=9, shape=(10,), dtype=np.int32)

        # Target divisor
        self.target_divisor = target_divisor

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_digits = np.zeros(10, dtype=np.int32)  # Array to store digits
        self.current_length = 0  # Current length of the cumulative number
        self.done = False
        self.current_player = 1  # Player 1 starts
        return self.cumulative_digits.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.cumulative_digits.copy(), -10, True, False, {}

        if action < 0 or action >= 9:
            # Invalid action
            self.done = True
            return self.cumulative_digits.copy(), -10, True, False, {}

        digit = action + 1  # Map action (0-8) to digit (1-9)

        # Append the digit to the cumulative number
        if self.current_length < 10:
            self.cumulative_digits[self.current_length] = digit
            self.current_length += 1
        else:
            # Maximum length reached, current player loses
            self.done = True
            reward = -1  # Loss
            return self.cumulative_digits.copy(), reward, True, False, {}

        # Build the cumulative number as an integer
        cumulative_number = int(
            "".join(map(str, self.cumulative_digits[: self.current_length]))
        )

        # Check if the cumulative number is divisible by the target divisor
        if cumulative_number % self.target_divisor == 0:
            # Current player wins
            self.done = True
            reward = 1
            return self.cumulative_digits.copy(), reward, True, False, {}

        # Check for maximum length loss condition
        if self.current_length == 10:
            # Current player loses
            self.done = True
            reward = -1  # Loss
            return self.cumulative_digits.copy(), reward, True, False, {}

        # Game continues
        reward = -10  # Penalty for valid move (per prompt)
        self.current_player = 3 - self.current_player  # Switch player (1 <-> 2)
        return self.cumulative_digits.copy(), reward, False, False, {}

    def render(self):
        # Create a string representation of the cumulative number
        cumulative_number_str = "".join(
            map(str, self.cumulative_digits[: self.current_length])
        )
        if cumulative_number_str == "":
            cumulative_number_str = "(empty)"
        output = f"Cumulative Number: {cumulative_number_str}\n"
        output += f"Current Player: Player {self.current_player}"
        return output

    def valid_moves(self):
        # All digits from 1 to 9 are always valid
        return list(range(9))
