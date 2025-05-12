import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Constants
        self.MAX_NUM_NUMBERS = 31  # Maximum length of the Shared List
        self.MAX_NUMBER = 31  # Maximum value allowed in the numbers
        self.MAX_ACTIONS = self.MAX_NUM_NUMBERS * self.MAX_NUMBER

        # Define action and observation space
        self.action_space = spaces.Discrete(self.MAX_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=self.MAX_NUMBER, shape=(self.MAX_NUM_NUMBERS,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.Shared_List = np.zeros(self.MAX_NUM_NUMBERS, dtype=np.int32)
        # Set Initial Number, default to 15 if not provided in options
        if options and "Initial_Number" in options:
            initial_number = options["Initial_Number"]
            if initial_number < 1 or initial_number > self.MAX_NUMBER:
                raise ValueError(
                    f"Initial_Number must be between 1 and {self.MAX_NUMBER}"
                )
            self.Shared_List[0] = int(initial_number)
        else:
            self.Shared_List[0] = 15  # Default Initial Number

        self.done = False
        return self.Shared_List.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.Shared_List.copy(), 0, True, False, {}

        position_in_list = action // self.MAX_NUMBER
        first_split_number = (
            action % self.MAX_NUMBER
        ) + 1  # Split numbers start from 1

        # Validate position_in_list
        if position_in_list >= self.MAX_NUM_NUMBERS:
            reward = -10
            self.done = True
            return self.Shared_List.copy(), reward, self.done, False, {}

        N = self.Shared_List[position_in_list]

        # Check if N > 1 (can be split)
        if N <= 1:
            reward = -10
            self.done = True
            return self.Shared_List.copy(), reward, self.done, False, {}

        # Validate first_split_number
        if first_split_number < 1 or first_split_number >= N:
            reward = -10
            self.done = True
            return self.Shared_List.copy(), reward, self.done, False, {}

        second_split_number = N - first_split_number

        if second_split_number < 1:
            reward = -10
            self.done = True
            return self.Shared_List.copy(), reward, self.done, False, {}

        # Valid move: Perform the split
        # Remove N from Shared_List
        self.Shared_List[position_in_list] = 0

        # Add the two new numbers to Shared_List
        empty_positions = np.where(self.Shared_List == 0)[0]
        if len(empty_positions) < 2:
            # Not enough space to add new numbers (should not happen)
            reward = -10
            self.done = True
            return self.Shared_List.copy(), reward, self.done, False, {}

        self.Shared_List[empty_positions[0]] = first_split_number
        self.Shared_List[empty_positions[1]] = second_split_number

        # Check if there are any valid moves left
        if not self._has_valid_moves():
            # Current player wins
            reward = 1
            self.done = True
        else:
            # Game continues
            reward = -10
            self.done = False

        return self.Shared_List.copy(), reward, self.done, False, {}

    def _has_valid_moves(self):
        # Check if there are any numbers greater than 1 in Shared_List
        return np.any(self.Shared_List > 1)

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for idx in range(len(self.Shared_List)):
            N = self.Shared_List[idx]
            if N > 1:
                # For each possible valid split of N
                for first_split_number in range(1, N):
                    action_index = idx * self.MAX_NUMBER + (first_split_number - 1)
                    valid_actions.append(action_index)
        return valid_actions

    def render(self):
        # Return a string representation of the Shared List
        shared_list_numbers = [n for n in self.Shared_List if n > 0]
        shared_list_str = "Shared List: " + str(shared_list_numbers)
        return shared_list_str
