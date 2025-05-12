import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define constants
        self.MAX_SEQUENCE_LENGTH = 15  # Maximum length of the sequence
        self.NUM_NUMBERS = 9  # Numbers from 1 to 9
        self.NUM_POSITIONS = 2  # Front or Back

        # Define action space: 9 numbers * 2 positions = 18 possible actions
        self.action_space = spaces.Discrete(self.NUM_NUMBERS * self.NUM_POSITIONS)

        # Observation space: sequence of numbers, padded with zeros if necessary
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.MAX_SEQUENCE_LENGTH,), dtype=np.int32
        )

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = []  # Shared sequence of numbers
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Returns the observation (sequence padded to MAX_SEQUENCE_LENGTH)
        obs = np.zeros(self.MAX_SEQUENCE_LENGTH, dtype=np.int32)
        seq_length = len(self.sequence)
        if seq_length > 0:
            obs[:seq_length] = self.sequence
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game already over

        # Map the action to a number and position
        number_index = action // self.NUM_POSITIONS
        position_index = action % self.NUM_POSITIONS
        number = number_index + 1  # Numbers from 1 to 9
        position = "front" if position_index == 0 else "back"

        # Add the number to the sequence
        if position == "front":
            self.sequence.insert(0, number)
        else:
            self.sequence.append(number)

        # Check if the sequence is a palindrome
        if self.is_palindrome(self.sequence):
            # Current player loses
            self.done = True
            reward = -1  # Negative reward for losing
            return self._get_obs(), reward, True, False, {}

        # Check if sequence exceeds maximum length (unlikely but safeguards the environment)
        if len(self.sequence) >= self.MAX_SEQUENCE_LENGTH:
            self.done = True
            reward = -1  # Negative reward for exceeding max length
            return self._get_obs(), reward, True, False, {}

        # Valid move, game continues
        reward = -10  # Negative reward for a valid move
        self.current_player = 3 - self.current_player  # Switch player between 1 and 2
        return self._get_obs(), reward, False, False, {}

    def render(self):
        # Returns a string representation of the current sequence
        if not self.sequence:
            return "Current Sequence: (empty)"
        else:
            return "Current Sequence: " + "-".join(map(str, self.sequence))

    def valid_moves(self):
        # All moves are always valid unless the game is over
        if self.done:
            return []
        return list(range(self.action_space.n))

    def is_palindrome(self, seq):
        # Returns True if the sequence is a palindrome
        return seq == seq[::-1]
