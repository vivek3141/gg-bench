import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Define action and observation space
        # There are 7 positions and 26 letters, so 7 * 26 = 182 possible actions
        self.action_space = spaces.Discrete(182)
        self.observation_space = spaces.Box(low=0, high=26, shape=(7,), dtype=np.int8)

        # Initialize the board and current player
        self.board = np.zeros(7, dtype=np.int8)
        self.current_player = 1  # Can be 1 or -1 to represent players
        self.done = False

        # Letter to number mapping
        self.number_to_letter = {i + 1: chr(ord("A") + i) for i in range(26)}
        self.letter_to_number = {chr(ord("A") + i): i + 1 for i in range(26)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(7, dtype=np.int8)
        self.current_player = 1
        self.done = False
        return self.board.copy(), {}

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Map action to position and letter
        action_index = action  # 0 to 181
        position = action_index // 26  # 0 to 6
        letter_index = action_index % 26  # 0 to 25
        letter_code = letter_index + 1  # 1 to 26

        # Check if the move is valid
        if self.board[position] != 0:
            reward = -10
            self.done = True
            return self.board.copy(), reward, True, False, {}

        # Place the letter on the board
        self.board[position] = letter_code

        # Check for palindrome
        if self.check_for_palindrome():
            reward = 1  # Current player wins
            self.done = True
            return self.board.copy(), reward, True, False, {}

        # Check for draw (board full without palindrome)
        if np.all(self.board != 0):
            reward = -1  # Current player loses
            self.done = True
            return self.board.copy(), reward, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        reward = 0
        return self.board.copy(), reward, False, False, {}

    def check_for_palindrome(self):
        # Check all substrings of length 3 to 7 for palindrome
        for start in range(7):
            for end in range(start + 2, 7):
                substring = self.board[start : end + 1]
                if 0 in substring:
                    continue
                if np.array_equal(substring, substring[::-1]):
                    return True
        return False

    def render(self):
        board_str = ""
        for i in range(7):
            if self.board[i] == 0:
                board_str += "_ "
            else:
                letter = self.number_to_letter[self.board[i]]
                board_str += f"{letter} "
        return f"Board: {board_str.strip()}"

    def valid_moves(self):
        valid_actions = []
        for position in range(7):
            if self.board[position] == 0:
                for letter_code in range(1, 27):
                    letter_index = letter_code - 1
                    action_index = position * 26 + letter_index
                    valid_actions.append(action_index)
        return valid_actions
