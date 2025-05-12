import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define word list: 5-letter words with no repeating letters
        # For simplicity, we use a small sample word list
        self.word_list = [
            "CRANE",
            "FAULT",
            "WATER",
            "BRING",
            "PLANT",
            "SMART",
            "GLIDE",
            "HONEY",
            "QUICK",
            "FROZE",
            "VAPID",
            "BLEND",
            "CHIRP",
            "DROPS",
            "FLAME",
            "GHOST",
            "HUMAN",
            "KINGS",
            "JELLY",
            "LIGHT",
            "MUSIC",
            "NIGHT",
            "OPENS",
            "PULSE",
            "QUEST",
            "RAISE",
            "SHINE",
            "TRICK",
            "UNITY",
            "VOWEL",
            "WRONG",
            "YEAST",
            "ZEBRA",
            "ACORN",
            "BLAZE",
            "CHARM",
            "DRIVE",
            "EAGLE",
            "FRAME",
            "GRIND",
            "HIPPO",
            "INPUT",
            "JUICE",
            "KRAUT",
            "LEMON",
            "MAJOR",
            "NURSE",
            "OCEAN",
            "PRIZE",
            "QUILT",
            "RUMOR",
            "SOLVE",
            "THANK",
            "USAGE",
            "VIOLA",
            "WHOLE",
            "YOUNG",
            "ZONED",
            "BLANK",
            "CHIEF",
            "DEPTH",
            "EXTRA",
            "FAITH",
            "GLOBE",
            "HEIST",
            "INDEX",
            "JOINT",
            "KNIFE",
            "LUNCH",
            "MIGHT",
            "NOISE",
            "OFFER",
            "POWER",
            "QUARK",
            "RHYME",
            "SOUTH",
            "THOSE",
            "URBAN",
            "VOICE",
            "WORLD",
            "XENON",
            "YACHT",
            "ADORE",
            "BRAVE",
            "CLOWN",
            "DAILY",
            "EVENT",
            "FROST",
            "GLEAM",
            "HEART",
            "IDEAL",
            "JOKER",
            "KOALA",
            "LAUGH",
            "MAGIC",
            "NERVE",
            "OWNER",
            "PRIME",
            "QUERY",
            "RANGE",
            "SHAPE",
            "TIGER",
            "ULTRA",
            "VIVID",
            "WITCH",
        ]  # Ensure words are 5 letters with no repeating letters and unique letters

        # Define action space
        # Actions 0-129: letter-position guesses (26 letters * 5 positions)
        # Actions 130 onwards: full word guesses
        self.action_space = spaces.Discrete(130 + len(self.word_list))

        # Observation space: shape (26,5)
        # 0: unknown, 1: absent, 2: present, 3: correct
        self.observation_space = spaces.Box(low=0, high=3, shape=(26, 5), dtype=np.int8)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Select code words for both players
        self.code_words = np.random.choice(self.word_list, size=2, replace=False)

        # Initialize knowledge bases for both players
        self.knowledge_base = [
            np.zeros((26, 5), dtype=np.int8),  # Player 0's knowledge base
            np.zeros((26, 5), dtype=np.int8),  # Player 1's knowledge base
        ]

        # Set current player
        self.current_player = 0  # Player 0 starts

        self.done = False
        return self.knowledge_base[self.current_player], {}

    def step(self, action):
        if self.done:
            # Game is over
            return self.knowledge_base[self.current_player], 0, True, False, {}

        kb = self.knowledge_base[self.current_player]
        opponent_code_word = self.code_words[1 - self.current_player]

        if action < 130:
            # Letter-position guess
            position_index = action // 26
            letter_index = action % 26
            letter = chr(ord("A") + letter_index)

            # Check for invalid move (repeating same letter-position guess)
            if kb[letter_index, position_index] != 0:
                # Invalid move
                reward = -10
                self.done = True
                return kb, reward, True, False, {}

            # Get feedback from opponent
            feedback = self.get_feedback(letter, position_index, opponent_code_word)

            # Update knowledge base
            if feedback == "Correct":
                kb[letter_index, position_index] = 3
            elif feedback == "Present":
                kb[letter_index, position_index] = 2
            elif feedback == "Absent":
                kb[letter_index, position_index] = 1

            # Check for win condition
            if self.check_win_condition(kb, opponent_code_word):
                reward = 1
                self.done = True
                return kb, reward, True, False, {}
            else:
                # No reward, switch to next player
                self.current_player = 1 - self.current_player
                return self.knowledge_base[self.current_player], 0, False, False, {}
        else:
            # Full word guess
            word_index = action - 130
            if word_index >= len(self.word_list):
                # Invalid action
                reward = -10
                self.done = True
                return kb, reward, True, False, {}

            guess_word = self.word_list[word_index]
            if guess_word == opponent_code_word:
                # Correct guess, player wins
                reward = 1
                self.done = True
            else:
                # Incorrect guess, game continues
                reward = 0
                # According to rules, turn passes to the opponent
                self.current_player = 1 - self.current_player

            return (
                self.knowledge_base[self.current_player],
                reward,
                self.done,
                False,
                {},
            )

    def get_feedback(self, letter, position_index, code_word):
        if letter not in code_word:
            return "Absent"
        elif code_word[position_index] == letter:
            return "Correct"
        else:
            return "Present"

    def check_win_condition(self, knowledge_base, opponent_code_word):
        for position_index in range(5):
            letter = opponent_code_word[position_index]
            letter_index = ord(letter) - ord("A")
            if knowledge_base[letter_index, position_index] != 3:
                return False
        return True

    def render(self):
        kb = self.knowledge_base[self.current_player]
        output = ""
        output += f"Player {self.current_player}'s Knowledge Base:\n"
        output += "       Pos1  Pos2  Pos3  Pos4  Pos5\n"
        for letter_index in range(26):
            letter = chr(ord("A") + letter_index)
            output += f"{letter}:    "
            for position_index in range(5):
                status = kb[letter_index, position_index]
                if status == 0:
                    output += "    . "
                elif status == 1:
                    output += " Absn "
                elif status == 2:
                    output += " Pres "
                elif status == 3:
                    output += " Corr "
            output += "\n"
        return output

    def valid_moves(self):
        kb = self.knowledge_base[self.current_player]
        valid_actions = []
        # Letter-position guesses
        for action in range(130):
            position_index = action // 26
            letter_index = action % 26
            if kb[letter_index, position_index] == 0:
                valid_actions.append(action)
        # Full word guesses
        for action in range(130, 130 + len(self.word_list)):
            valid_actions.append(action)
        return valid_actions
