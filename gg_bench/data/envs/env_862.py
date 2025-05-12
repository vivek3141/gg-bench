import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: digits 1 to 9 mapped to actions 0 to 8
        self.action_space = spaces.Discrete(9)

        # Define observation space
        # Observation consists of:
        # - shared number represented by up to 9 digits (positions 0-8)
        # - available digits (positions 9-17), 1 if available, 0 if used
        # - current player (position 18), 1 or 2
        # All values are integers between 0 and 9
        self.observation_space = spaces.Box(low=0, high=9, shape=(19,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number_digits = [0] * 9  # Up to 9 digits
        self.shared_number = 0
        self.available_digits = [1] * 9  # Digits 1-9 are available
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.turns_taken = 0
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}
        if action < 0 or action > 8:
            return self._get_observation(), -10, True, False, {}
        if self.available_digits[action] == 0:
            return self._get_observation(), -10, True, False, {}

        # Valid action
        selected_digit = action + 1  # Map action to digit
        self.available_digits[action] = 0  # Mark digit as used

        # Append digit to shared number
        self.shared_number_digits[self.turns_taken] = selected_digit
        self.shared_number = self.shared_number * 10 + selected_digit
        self.turns_taken += 1

        # Check for win
        if self.shared_number % 7 == 0:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Check if all digits are used
        if self.turns_taken == 9:
            self.done = True
            # Fallback Victory
            distance_player = self.shared_number % 7
            distance_opponent = (
                self.shared_number % 7
            )  # Since same number, distances are equal
            # In case of tie, last player to have moved loses
            reward = -1  # Current player loses in tie
            return self._get_observation(), reward, True, False, {}

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1
        return self._get_observation(), -10, False, False, {}

    def render(self):
        shared_number_str = "".join(str(d) for d in self.shared_number_digits if d != 0)
        available_digits_str = " ".join(
            str(i + 1) for i in range(9) if self.available_digits[i] == 1
        )
        state_str = f"Shared Number: {shared_number_str}\n"
        state_str += f"Available Digits: {available_digits_str}\n"
        state_str += f"Current Player: Player {self.current_player}\n"
        return state_str

    def valid_moves(self):
        return [i for i in range(9) if self.available_digits[i] == 1]

    def _get_observation(self):
        observation = np.array(
            self.shared_number_digits + self.available_digits + [self.current_player],
            dtype=np.int8,
        )
        return observation
