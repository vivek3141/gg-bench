import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_word="CODE"):
        super(CustomEnv, self).__init__()

        # Convert the target word to indices (0-25)
        self.target_word_str = target_word.upper()
        self.target_word = np.array(
            [ord(char) - ord("A") for char in self.target_word_str], dtype=np.int32
        )
        self.word_length = len(self.target_word)

        # Define action and observation space
        # Actions correspond to positions in the word to increment (0 to word_length - 1)
        self.action_space = spaces.Discrete(self.word_length)

        # Observation is the concatenation of current_word and target_word indices
        # Each letter is represented by an integer from 0 ('A') to 25 ('Z')
        self.observation_space = spaces.MultiDiscrete([26] * (2 * self.word_length))

        # Initialize variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Current word starts with all 'A's, represented by zeros
        self.current_word = np.zeros(self.word_length, dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        # Observation is the concatenation of current_word and target_word
        observation = np.concatenate((self.current_word, self.target_word))
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if action is valid
        if not (0 <= action < self.word_length):
            # Invalid action (position out of range)
            return (
                np.concatenate((self.current_word, self.target_word)),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        if self.done:
            # Game is already over
            return (
                np.concatenate((self.current_word, self.target_word)),
                0,
                True,
                False,
                {},
            )

        # Increment the letter at the chosen position
        self.current_word[action] = (
            self.current_word[action] + 1
        ) % 26  # Wrap around at 26 ('Z'+1 -> 'A')

        # Check for win condition
        if np.array_equal(self.current_word, self.target_word):
            self.done = True
            reward = 1  # Current player wins
            return (
                np.concatenate((self.current_word, self.target_word)),
                reward,
                True,
                False,
                {},
            )
        else:
            # Continue the game
            reward = 0
            # Switch player
            self.current_player = 1 if self.current_player == 2 else 2
            return (
                np.concatenate((self.current_word, self.target_word)),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        # Convert indices back to letters
        current_word_letters = [chr(idx + ord("A")) for idx in self.current_word]
        target_word_letters = [chr(idx + ord("A")) for idx in self.target_word]
        render_str = "Target Word : " + " ".join(target_word_letters) + "\n"
        render_str += "Current Word: " + " ".join(current_word_letters) + "\n"
        render_str += f"Player {self.current_player}'s turn\n"
        return render_str

    def valid_moves(self):
        # All positions are valid moves (0 to word_length - 1)
        return list(range(self.word_length))
