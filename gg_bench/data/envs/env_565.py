import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=16):
        super(CustomEnv, self).__init__()

        # Constants
        self.MAX_NUMBER = 100  # Maximum number value in the game
        self.MAX_LENGTH = 50  # Maximum length of the numbers list

        # Action and observation spaces
        self.action_space = spaces.Discrete(self.MAX_LENGTH * (self.MAX_NUMBER - 1))
        self.observation_space = spaces.Box(
            low=0, high=self.MAX_NUMBER, shape=(self.MAX_LENGTH,), dtype=np.int32
        )

        # Initialize game state
        self.starting_number = starting_number
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the numbers list with the starting number
        self.numbers_list = [self.starting_number]
        self.current_player = 1  # Player 1 starts
        self.done = False
        self._update_observation()
        return self.observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.observation, 0, True, False, {}

        # Decode action into number index and split value
        number_index = action // (self.MAX_NUMBER - 1)
        split_value = (action % (self.MAX_NUMBER - 1)) + 1

        # Check if action is valid
        if number_index >= len(self.numbers_list):
            # Invalid number index
            return self.observation, -10, True, False, {}

        number = self.numbers_list[number_index]
        if number <= 1:
            # Cannot split a number less than or equal to 1
            return self.observation, -10, True, False, {}

        if split_value < 1 or split_value >= number:
            # Invalid split value
            return self.observation, -10, True, False, {}

        # Perform the split
        self.numbers_list.pop(number_index)
        self.numbers_list.append(split_value)
        self.numbers_list.append(number - split_value)

        # Truncate numbers list if it exceeds MAX_LENGTH
        if len(self.numbers_list) > self.MAX_LENGTH:
            self.numbers_list = self.numbers_list[: self.MAX_LENGTH]

        # Update observation
        self._update_observation()

        # Check if the game is over (opponent cannot make a move)
        game_over = all(num <= 1 for num in self.numbers_list)
        if game_over:
            self.done = True
            return self.observation, 1, True, False, {}
        else:
            # Switch current player
            self.current_player = 2 if self.current_player == 1 else 1
            return self.observation, 0, False, False, {}

    def render(self):
        # Returns a string representation of the current game state
        numbers_str = ", ".join(map(str, self.numbers_list))
        return f"Current Player: Player {self.current_player}\nNumbers List: [{numbers_str}]"

    def valid_moves(self):
        valid_actions = []
        for idx, num in enumerate(self.numbers_list):
            if num > 1:
                for split_val in range(1, num):
                    action_index = idx * (self.MAX_NUMBER - 1) + (split_val - 1)
                    valid_actions.append(action_index)
        return valid_actions

    def _update_observation(self):
        # Pad the numbers list to match MAX_LENGTH
        padded_numbers = self.numbers_list + [0] * (
            self.MAX_LENGTH - len(self.numbers_list)
        )
        self.observation = np.array(padded_numbers, dtype=np.int32)
