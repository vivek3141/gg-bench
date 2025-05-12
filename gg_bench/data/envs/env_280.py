import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0:'A', 1:'B', 2:'C'
        self.action_space = spaces.Discrete(3)

        # Observations: 0: empty, 1:'A', 2:'B', 3:'C'
        # The string can have up to 5 characters
        self.observation_space = spaces.MultiDiscrete([4, 4, 4, 4, 4])

        # Mapping dictionaries
        self.action_to_char = {0: "A", 1: "B", 2: "C"}
        self.char_to_obs = {"A": 1, "B": 2, "C": 3}
        self.obs_to_char = {0: "", 1: "A", 2: "B", 3: "C"}

        # Forbidden substring (default "AB")
        self.forbidden_substring = "AB"

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the current string as an empty observation array
        self.current_string = np.zeros(5, dtype=np.int8)
        self.string_length = 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.current_string, {}  # Observation and info

    def step(self, action):
        # Check if the game is already done
        if self.done:
            return (
                self.current_string,
                -10,
                True,
                False,
                {},
            )  # Invalid move after game over

        # Map action to character and observation value
        if action not in [0, 1, 2]:
            return self.current_string, -10, True, False, {}  # Invalid action
        char = self.action_to_char[action]
        obs_value = self.char_to_obs[char]

        # Check if the string is already at maximum length
        if self.string_length >= 5:
            self.done = True
            return (
                self.current_string,
                -10,
                True,
                False,
                {},
            )  # Invalid move, string is full

        # Append the character to the current string
        self.current_string[self.string_length] = obs_value
        self.string_length += 1

        # Check for forbidden substring
        current_chars = [
            self.obs_to_char[val] for val in self.current_string[: self.string_length]
        ]
        current_string_str = "".join(current_chars)
        if self.forbidden_substring in current_string_str:
            self.done = True
            return (
                self.current_string,
                -10,
                True,
                False,
                {},
            )  # Loss due to forbidden substring

        # Check if the string has reached five characters without forbidden substring
        if self.string_length == 5:
            self.done = True
            return (
                self.current_string,
                1,
                True,
                False,
                {},
            )  # Win by completing the string

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1
        return self.current_string, 0, False, False, {}  # Game continues

    def render(self):
        current_chars = [
            self.obs_to_char[val] for val in self.current_string[: self.string_length]
        ]
        current_string_str = "".join(current_chars)
        return f'Current String: "{current_string_str}"'

    def valid_moves(self):
        if self.done:
            return []  # No valid moves if the game is over
        return [0, 1, 2]  # Actions 0:'A', 1:'B', 2:'C'
