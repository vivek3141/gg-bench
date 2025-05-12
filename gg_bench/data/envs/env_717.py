import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The board is a 4x4 grid, flattened to a 16-element array
        # Each cell can have the following values:
        # 0: Empty and unblocked
        # -1: Blocked
        # 1: Marked by Player 1 ('X')
        # 2: Marked by Player 2 ('O')
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(16,), dtype=np.int8)

        # Initialize the board and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board to empty and unblocked
        self.board = np.zeros(16, dtype=np.int8)
        # Player 1 ('X') starts first
        self.current_player = 1
        # Game not over
        self.done = False
        # Return initial observation and info
        return self.board.copy(), {}

    def step(self, action):
        if self.done:
            # If the game is already over, ignore further moves
            return self.board.copy(), -10, True, False, {}

        if not self.is_valid_move(action):
            # Invalid move - cell is not empty and unblocked
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Place the current player's symbol on the selected square
        self.board[action] = self.current_player

        # Block adjacent squares
        self.block_adjacent_squares(action)

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if the next player has any valid moves
        if not self.has_valid_moves():
            # Current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Game continues
        return self.board.copy(), -10, False, False, {}

    def render(self):
        # Return a string representation of the board
        board_str = ""
        for i in range(4):
            for j in range(4):
                cell = self.board[i * 4 + j]
                if cell == 0:
                    board_str += ". "
                elif cell == -1:
                    board_str += "# "
                elif cell == 1:
                    board_str += "X "
                elif cell == 2:
                    board_str += "O "
            board_str += "\n"
        return board_str

    def valid_moves(self):
        # Return a list of integers representing valid moves
        return [i for i in range(16) if self.board[i] == 0]

    # Helper methods
    def is_valid_move(self, action):
        # A move is valid if the selected square is empty and unblocked
        return self.board[action] == 0

    def block_adjacent_squares(self, action):
        # Block all adjacent squares to the selected square
        row = action // 4
        col = action % 4
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r = row + dr
                c = col + dc
                if 0 <= r < 4 and 0 <= c < 4:
                    idx = r * 4 + c
                    if self.board[idx] == 0 and not (r == row and c == col):
                        # Block the square if it's empty and not the selected square
                        self.board[idx] = -1

    def has_valid_moves(self):
        # Check if the next player has any valid moves
        return any(self.board[i] == 0 for i in range(16))
