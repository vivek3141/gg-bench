import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        # Actions 0-8: Change tens digit to 1-9
        # Actions 9-18: Change ones digit to 0-9
        self.action_space = spaces.Discrete(19)

        # Define action mapping
        self.action_map = {}
        # Actions for changing tens digit (digit index 0)
        for i in range(9):  # i from 0 to 8
            action_index = i
            digit_index = 0  # tens digit
            new_digit_value = i + 1  # digits 1..9
            self.action_map[action_index] = (digit_index, new_digit_value)
        # Actions for changing ones digit (digit index 1)
        for i in range(10):  # i from 0 to 9
            action_index = i + 9
            digit_index = 1  # ones digit
            new_digit_value = i  # digits 0..9
            self.action_map[action_index] = (digit_index, new_digit_value)

        # Define observation space
        # Observation: [current_number, current_player]
        # current_number: 10 to 99
        # current_player: 1 or 2
        self.observation_space = spaces.Box(
            low=np.array([10, 1]), high=np.array([99, 2]), dtype=np.int32
        )

        self.current_number = None
        self.current_player = None
        self.done = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random starting number between 10 and 99, not a multiple of 7
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        while True:
            self.current_number = self.np_random.randint(10, 100)
            if self.current_number % 7 != 0:
                break
        self.current_player = 1  # Start with player 1
        self.done = False
        observation = np.array(
            [self.current_number, self.current_player], dtype=np.int32
        )
        info = {}
        return observation, info  # observation, info(dict)

    def step(self, action):
        if self.done:
            # Game is over
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, 0, True, False, info

        # Check if action is valid
        if action not in self.action_map:
            # Invalid action index
            reward = -10
            self.done = True  # End the game due to invalid action
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, reward, self.done, False, info

        # Map action to (digit_index, new_digit_value)
        digit_index, new_digit_value = self.action_map[action]
        digits = [int(d) for d in str(self.current_number)]

        # Ensure digits has length 2
        if len(digits) != 2:
            # Should not happen, but handle just in case
            reward = -10
            self.done = True
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, reward, self.done, False, info

        # Copy digits to new list
        new_digits = digits.copy()
        # Change selected digit
        new_digits[digit_index] = new_digit_value

        # Validate the new number
        # Tens digit cannot be zero
        if new_digits[0] == 0:
            # Invalid move
            reward = -10
            # Forfeit turn, switch player
            self.current_player = 2 if self.current_player == 1 else 1
            self.done = False
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, reward, self.done, False, info

        # Construct new number
        new_number = new_digits[0] * 10 + new_digits[1]

        # Check if new number is valid (between 10 and 99)
        if new_number < 10 or new_number > 99:
            # Invalid move
            reward = -10
            # Forfeit turn, switch player
            self.current_player = 2 if self.current_player == 1 else 1
            self.done = False
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, reward, self.done, False, info

        # Check if the move actually changes the number
        if new_number == self.current_number:
            # Invalid move (no change)
            reward = -10
            # Forfeit turn, switch player
            self.current_player = 2 if self.current_player == 1 else 1
            self.done = False
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, reward, self.done, False, info

        # Valid move
        self.current_number = new_number

        # Check if new number is a multiple of 7
        if self.current_number % 7 == 0:
            # Current player wins
            reward = 1
            self.done = True
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, reward, self.done, False, info
        else:
            # Game continues
            reward = 0
            # Switch player
            self.current_player = 2 if self.current_player == 1 else 1
            observation = np.array(
                [self.current_number, self.current_player], dtype=np.int32
            )
            info = {}
            return observation, reward, self.done, False, info

    def render(self):
        return f"Current number: {self.current_number}, Current player: Player {self.current_player}"

    def valid_moves(self):
        valid_actions = []
        digits = [int(d) for d in str(self.current_number)]
        for action, (digit_index, new_digit_value) in self.action_map.items():
            new_digits = digits.copy()
            new_digits[digit_index] = new_digit_value
            # Tens digit cannot be zero
            if new_digits[0] == 0:
                continue
            new_number = new_digits[0] * 10 + new_digits[1]
            # Number must be between 10 and 99 and different from current number
            if 10 <= new_number <= 99 and new_number != self.current_number:
                valid_actions.append(action)
        return valid_actions
