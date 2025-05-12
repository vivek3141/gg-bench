import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define word length
        self.word_length = 4  # For example, words of length 4

        # Load dictionary of valid words of the specified length
        self.dictionary = self.load_dictionary(self.word_length)

        # Define action and observation space
        # Action space: L positions * 26 letters
        self.action_space = spaces.Discrete(self.word_length * 26)

        # Observation space: shared word and player's target word, both encoded as integers [0-25]
        self.observation_space = spaces.Box(
            low=0, high=25, shape=(self.word_length * 2,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the shared word
        self.shared_word = "COLD"  # Starting word
        self.shared_word_indices = self.word_to_indices(self.shared_word)
        self.previous_words = [self.shared_word]

        # Secret target words for players (could be randomized)
        self.player1_target_word = "WARM"
        self.player1_target_indices = self.word_to_indices(self.player1_target_word)

        self.player2_target_word = "HEAT"
        self.player2_target_indices = self.word_to_indices(self.player2_target_word)

        # Current player (1 or 2)
        self.current_player = 1

        # Game status
        self.done = False

        # Prepare the initial observation
        observation = self.get_observation()
        info = {}
        return observation, info

    def step(self, action):
        if self.done:
            # If the game is over, no further actions are valid
            reward = 0
            return self.get_observation(), reward, self.done, False, {}

        # Decode the action into position and new letter
        position = action // 26
        new_letter_index = action % 26

        # Validate position
        if position < 0 or position >= self.word_length:
            reward = -10
            self.done = True
            return self.get_observation(), reward, self.done, False, {}

        # Create the new word by changing one letter
        new_word_indices = self.shared_word_indices.copy()
        old_letter_index = new_word_indices[position]

        # Check if the letter actually changes
        if old_letter_index == new_letter_index:
            reward = -10
            self.done = True
            return self.get_observation(), reward, self.done, False, {}

        new_word_indices[position] = new_letter_index
        new_word = self.indices_to_word(new_word_indices)

        # Check if new word is valid
        if new_word not in self.dictionary:
            reward = -10
            self.done = True
            return self.get_observation(), reward, self.done, False, {}

        # Check if new word has been used before
        if new_word in self.previous_words:
            reward = -10
            self.done = True
            return self.get_observation(), reward, self.done, False, {}

        # Update the shared word
        self.shared_word = new_word
        self.shared_word_indices = new_word_indices
        self.previous_words.append(new_word)

        # Check for victory
        if self.current_player == 1:
            target_indices = self.player1_target_indices
        else:
            target_indices = self.player2_target_indices

        if np.array_equal(self.shared_word_indices, target_indices):
            reward = 1
            self.done = True
            return self.get_observation(), reward, self.done, False, {}

        # Check for stalemate: both players pass consecutively (not implemented in this example)
        # For simplicity, we assume the game continues until a player wins or makes an invalid move

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Return the observation
        reward = 0  # No reward for a regular move
        return self.get_observation(), reward, self.done, False, {}

    def render(self):
        # Returns a string representing the current state of the game
        s = f"Current Player: Player {self.current_player}\n"
        s += f"Shared Word: {self.shared_word}\n"
        if self.current_player == 1:
            s += f"Your Target Word: {self.player1_target_word}\n"
        else:
            s += f"Your Target Word: {self.player2_target_word}\n"
        return s

    def valid_moves(self):
        # Returns a list of valid actions (integers)
        valid_actions = []
        for position in range(self.word_length):
            for letter_index in range(26):
                if letter_index != self.shared_word_indices[position]:
                    # Create new word
                    new_word_indices = self.shared_word_indices.copy()
                    new_word_indices[position] = letter_index
                    new_word = self.indices_to_word(new_word_indices)
                    # Check if the new word is valid and not previously used
                    if (
                        new_word in self.dictionary
                        and new_word not in self.previous_words
                    ):
                        action = position * 26 + letter_index
                        valid_actions.append(action)
        return valid_actions

    def get_observation(self):
        # Prepare the observation array
        if self.current_player == 1:
            target_indices = self.player1_target_indices
        else:
            target_indices = self.player2_target_indices

        observation = np.concatenate((self.shared_word_indices, target_indices))
        return observation

    @staticmethod
    def word_to_indices(word):
        # Converts a word into an array of indices (integers between 0 and 25)
        return np.array([ord(c) - ord("A") for c in word.upper()], dtype=np.int32)

    @staticmethod
    def indices_to_word(indices):
        # Converts an array of indices into a word
        return "".join([chr(i + ord("A")) for i in indices])

    @staticmethod
    def load_dictionary(word_length):
        # For simplicity, a small predefined set of valid words of the specified length
        valid_words = {
            4: {
                "COLD",
                "CORD",
                "CARD",
                "WARD",
                "WARM",
                "HEAT",
                "HEAD",
                "HARD",
                "HARE",
                "WARE",
                "WARS",
                "WART",
                "HART",
                "HARD",
                "WORD",
                "WOLD",
                "WORD",
                "WORM",
                "WORN",
                "HORN",
                "HURT",
                "HUNT",
                "HEAL",
                "HEEL",
                "HELL",
                "HELP",
                "HERD",
                "HERB",
                "HERB",
                "HERD",
                "HERE",
                "HERO",
            },
            # Add more word lengths and words as needed
        }
        return valid_words.get(word_length, set())

    def seed(self, seed=None):
        # Set the random seed for reproducibility (optional)
        np.random.seed(seed)
