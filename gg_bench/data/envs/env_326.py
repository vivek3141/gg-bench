import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define constants
        self.MAX_SEQUENCE_LENGTH = 21  # Maximum length of the sequence
        self.NUM_LETTERS = 26  # Number of letters in the alphabet
        self.ACTION_SPACE_SIZE = self.NUM_LETTERS * 2  # 26 letters * 2 positions

        # Define action and observation space
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(
            low=0,
            high=self.NUM_LETTERS - 1,  # Letters indexed from 0 to 25
            shape=(self.MAX_SEQUENCE_LENGTH,),
            dtype=np.int8,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = []  # Current sequence of letters as indices [0-25]
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.winner = None
        self.info = {}
        # Observation is the sequence padded to MAX_SEQUENCE_LENGTH
        observation = self._get_observation()
        return observation, self.info  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over; return appropriate observation and penalty
            observation = self._get_observation()
            return observation, -10, True, False, self.info

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            observation = self._get_observation()
            self.done = True
            self.winner = -self.current_player  # Opponent wins
            return observation, -10, True, False, self.info

        # Map action to letter and position
        letter_index = action // 2  # 0 to 25
        position = action % 2  # 0 for beginning, 1 for end
        letter = chr(ord("a") + letter_index)

        # Add letter to sequence at the chosen position
        if position == 0:
            self.sequence.insert(0, letter_index)
        else:
            self.sequence.append(letter_index)

        # Check for palindrome
        if self._check_palindrome():
            self.done = True
            self.winner = self.current_player
            observation = self._get_observation()
            return observation, 1, True, False, self.info

        # No palindrome formed; switch player
        self.current_player *= -1
        observation = self._get_observation()
        return observation, 0, False, False, self.info

    def render(self):
        # Convert sequence of indices to letters for display
        sequence_letters = [chr(ord("a") + idx) for idx in self.sequence]
        sequence_str = " ".join(sequence_letters)
        result = f"Current Sequence: {sequence_str}"
        if self.done:
            if self.winner == 1:
                result += "\nPlayer 1 wins!"
            elif self.winner == -1:
                result += "\nPlayer 2 wins!"
            else:
                result += "\nGame over!"
        return result

    def valid_moves(self):
        if self.done:
            return []  # No valid moves if the game is over
        else:
            # All actions are valid unless the game is over
            return list(range(self.ACTION_SPACE_SIZE))

    def _check_palindrome(self):
        # Check for palindrome of length 5 or more in the sequence
        seq_length = len(self.sequence)
        if seq_length < 5:
            return False
        # Convert sequence indices to letters
        seq_str = "".join(chr(ord("a") + idx) for idx in self.sequence)
        # Check all substrings of length >=5 for palindrome
        for start in range(seq_length - 4):
            for end in range(start + 5, seq_length + 1):
                substring = seq_str[start:end]
                if substring == substring[::-1]:
                    return True
        return False

    def _get_observation(self):
        # Pad the sequence to MAX_SEQUENCE_LENGTH with -1
        pad_length = self.MAX_SEQUENCE_LENGTH - len(self.sequence)
        padded_sequence = self.sequence + [-1] * pad_length
        observation = np.array(
            padded_sequence[: self.MAX_SEQUENCE_LENGTH], dtype=np.int8
        )
        return observation
