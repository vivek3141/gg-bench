import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(26)

        # Observation space shape:
        # [Letter pool (26), Player 0 collection (26), Player 1 collection (26),
        #  Player 0 word letters (26), Player 1 word letters (26), current_player (1)]
        self.observation_space = spaces.Box(low=0, high=1, shape=(131,), dtype=int)

        # Predefined list of words with five unique letters
        self.words_list = ["BRAVE", "CHARM", "GHOST", "PLANT", "QUICK"]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the letter pool: 1 if available, 0 if taken
        self.letter_pool = np.ones(26, dtype=int)

        # Initialize players' collections
        self.player_collections = {
            0: np.zeros(26, dtype=int),  # Player 0
            1: np.zeros(26, dtype=int),  # Player 1
        }

        # Randomly select words for each player
        self.player_words = {
            0: self.word_to_indices(np.random.choice(self.words_list)),
            1: self.word_to_indices(np.random.choice(self.words_list)),
        }

        # Record the order in which players collected letters (for tie-breaker)
        self.player_letter_order = {0: [], 1: []}

        self.current_player = 0  # Player 0 starts
        self.done = False

        # Prepare initial observation
        observation = self.get_observation()

        return observation, {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.done:
            # Game is over
            return self.get_observation(), 0, True, False, info

        if action < 0 or action >= 26 or self.letter_pool[action] == 0:
            # Invalid action
            reward = -10
            terminated = True
            self.done = True
            return self.get_observation(), reward, terminated, truncated, info

        # Valid move
        # Remove letter from letter pool
        self.letter_pool[action] = 0
        # Add letter to player's collection
        self.player_collections[self.current_player][action] = 1
        # Record the order of letters collected (for tie-breaker)
        self.player_letter_order[self.current_player].append(action)

        # Check if the player has collected all letters of their word
        if self.check_player_wins(self.current_player):
            reward = 1
            terminated = True
            self.done = True
            return self.get_observation(), reward, terminated, truncated, info

        # Check if the letter pool is exhausted
        if np.sum(self.letter_pool) == 0:
            # Determine winner based on letters collected from their words
            winner = self.determine_winner()
            if winner == self.current_player:
                reward = 1
            else:
                reward = 0
            terminated = True
            self.done = True
            return self.get_observation(), reward, terminated, truncated, info

        # Switch to the other player
        self.current_player = 1 - self.current_player

        # Generate the new observation
        observation = self.get_observation()

        return observation, reward, terminated, truncated, info

    def render(self):
        # Create a string representation of the game state
        letter_pool_str = "Letter Pool: " + " ".join(
            [chr(i + ord("A")) if self.letter_pool[i] == 1 else "_" for i in range(26)]
        )
        player0_collection_str = "Player 1 Collection: " + " ".join(
            [chr(i + ord("A")) for i in range(26) if self.player_collections[0][i] == 1]
        )
        player1_collection_str = "Player 2 Collection: " + " ".join(
            [chr(i + ord("A")) for i in range(26) if self.player_collections[1][i] == 1]
        )
        current_player_str = f"Current Player: Player {self.current_player + 1}"
        return f"{letter_pool_str}\n{player0_collection_str}\n{player1_collection_str}\n{current_player_str}"

    def valid_moves(self):
        return [i for i in range(26) if self.letter_pool[i] == 1]

    def get_observation(self):
        # Construct the observation vector
        observation = np.concatenate(
            [
                self.letter_pool,
                self.player_collections[0],
                self.player_collections[1],
                self.player_words[0],
                self.player_words[1],
                np.array([self.current_player]),
            ]
        )
        return observation

    def word_to_indices(self, word):
        # Convert a word into a vector of letter indices
        indices = np.zeros(26, dtype=int)
        for char in word:
            index = ord(char.upper()) - ord("A")
            indices[index] = 1
        return indices

    def check_player_wins(self, player):
        # Check if the player has collected all letters of their word
        player_letters = self.player_words[player]
        collected_letters = self.player_collections[player]
        if np.all(player_letters <= collected_letters):
            return True
        return False

    def determine_winner(self):
        # Determine the winner based on letters collected from their words
        player0_letters = self.player_words[0]
        player1_letters = self.player_words[1]
        player0_collected = self.player_collections[0]
        player1_collected = self.player_collections[1]
        player0_score = np.sum(player0_letters * player0_collected)
        player1_score = np.sum(player1_letters * player1_collected)

        if player0_score > player1_score:
            return 0
        elif player1_score > player0_score:
            return 1
        else:
            # Tie-breaker: player who collected the first letter wins
            if not self.player_letter_order[0] and not self.player_letter_order[1]:
                return -1  # No letters collected
            elif not self.player_letter_order[0]:
                return 1
            elif not self.player_letter_order[1]:
                return 0
            else:
                if self.player_letter_order[0][0] < self.player_letter_order[1][0]:
                    return 0
                else:
                    return 1
