import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Game parameters
        self.target_word_length = 5  # Standard word length
        self.total_letters = 26  # Letters A-Z
        self.total_boards = 2  # Self and opponent
        self.total_slots = self.target_word_length  # Slots per board
        self.total_actions = (
            self.total_letters * self.total_boards * self.total_slots
        )  # Total possible actions

        # Define action and observation space
        self.action_space = spaces.Discrete(self.total_actions)
        self.observation_space = spaces.Box(
            low=0,
            high=self.total_letters,
            shape=(self.total_boards, self.total_slots),
            dtype=np.int32,
        )

        # Valid words (for simplicity, a small hardcoded set)
        self.valid_words = {
            "APPLE",
            "HOUSE",
            "OTHER",
            "THING",
            "WORDS",
            "FIRST",
            "HELLO",
            "THERE",
            "GAMES",
        }

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Boards: [0]: current player's board, [1]: opponent's board
        self.boards = np.zeros((self.total_boards, self.total_slots), dtype=np.int32)
        self.current_player = 0  # 0 or 1
        self.done = False
        return self.boards.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.boards.copy(), 0, True, False, {}

        # Decode the action
        letter_index, board_index, slot_index = self.decode_action(action)
        letter = letter_index + 1  # Letters are numbered from 1-26

        # Adjust board_index to actual boards
        if board_index == 0:
            board = self.boards[0]  # Self board
        else:
            board = self.boards[1]  # Opponent's board

        # Check if the slot is empty
        if board[slot_index] != 0:
            # Invalid move
            self.done = True
            reward = -10
            return self.boards.copy(), reward, True, False, {}

        # Place the letter
        board[slot_index] = letter

        # Check for board completion
        if np.all(board != 0):
            # Board is complete
            word = "".join(
                [chr(ord("A") + int(l - 1)) for l in board]
            )  # Convert letters to word
            if word in self.valid_words:
                # Word is valid
                if board_index == 0:
                    # Current player wins
                    self.done = True
                    reward = 1
                    return self.boards.copy(), reward, True, False, {}
                else:
                    # Current player loses
                    self.done = True
                    reward = -1
                    return self.boards.copy(), reward, True, False, {}
            else:
                # Word is invalid
                if board_index == 0:
                    # Current player loses
                    self.done = True
                    reward = -1
                    return self.boards.copy(), reward, True, False, {}
                else:
                    # Current player wins
                    self.done = True
                    reward = 1
                    return self.boards.copy(), reward, True, False, {}

        # Switch current player
        self.current_player = 1 - self.current_player
        # Swap boards to represent current player's perspective
        self.boards = np.flip(self.boards, axis=0)
        reward = 0
        return self.boards.copy(), reward, False, False, {}

    def render(self):
        def board_str(board):
            return " ".join(
                [chr(ord("A") + int(l - 1)) if l != 0 else "_" for l in board]
            )

        output = f"Your Board: [{board_str(self.boards[0])}]\n"
        output += f"Opponent's Board: [{board_str(self.boards[1])}]\n"
        return output

    def valid_moves(self):
        valid_actions = []
        for letter_index in range(self.total_letters):
            for board_index in range(self.total_boards):
                for slot_index in range(self.total_slots):
                    # Adjust board_index to actual boards
                    if board_index == 0:
                        board = self.boards[0]  # Self board
                    else:
                        board = self.boards[1]  # Opponent's board
                    if board[slot_index] == 0:
                        action = self.encode_action(
                            letter_index, board_index, slot_index
                        )
                        valid_actions.append(action)
        return valid_actions

    def encode_action(self, letter_index, board_index, slot_index):
        return (
            letter_index * (self.total_boards * self.total_slots)
            + board_index * self.total_slots
            + slot_index
        )

    def decode_action(self, action):
        total_actions_per_letter = self.total_boards * self.total_slots
        letter_index = action // total_actions_per_letter
        remainder = action % total_actions_per_letter
        board_index = remainder // self.total_slots
        slot_index = remainder % self.total_slots
        return letter_index, board_index, slot_index
