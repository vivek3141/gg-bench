import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(16), representing the 16 cells in the 4x4 grid
        self.action_space = spaces.Discrete(16)
        # The observation space is a 4x4 grid with values -1 (O), 0 (empty), or 1 (X)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4, 4), dtype=np.int8)

        # Initialize the board and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the 4x4 grid to zeros
        self.board = np.zeros((4, 4), dtype=np.int8)
        self.current_player = 1  # Player 1 uses 'X' (represented by 1)
        self.last_move = None  # No last move at the beginning
        self.done = False  # Game is not over
        return self.board.copy(), {}  # Return observation and empty info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self.board.copy(), 0, True, False, {}  # Game is over

        # Check if the action is valid
        if not self.is_valid_action(action):
            self.done = True
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Place the marker on the board
        row = action // 4
        col = action % 4
        self.board[row, col] = self.current_player
        self.last_move = (row, col)

        # Check if the opponent has any valid moves
        opponent_valid_moves = self.get_valid_moves_for_last_move(self.last_move)
        if not opponent_valid_moves:
            # Opponent cannot move; current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        return self.board.copy(), 0, False, False, {}  # Continue the game

    def render(self):
        # Create a string representation of the board
        board_str = ""
        for row in range(4):
            board_str += "|"
            for col in range(4):
                cell = self.board[row, col]
                if cell == 1:
                    board_str += " X |"
                elif cell == -1:
                    board_str += " O |"
                else:
                    board_str += "   |"
            board_str += "\n-----------------------------\n"
        return board_str

    def valid_moves(self):
        # Return the list of valid moves for the current player
        return self.get_valid_moves_for_last_move(self.last_move)

    def is_valid_action(self, action):
        row = action // 4
        col = action % 4
        # The cell must be empty
        if self.board[row, col] != 0:
            return False
        # On the first move, any empty cell is valid
        if self.last_move is None:
            return True
        # Check if the move is adjacent to the last move
        last_row, last_col = self.last_move
        if (abs(row - last_row) == 1 and col == last_col) or (
            abs(col - last_col) == 1 and row == last_row
        ):
            return True
        else:
            return False

    def get_valid_moves_for_last_move(self, last_move):
        valid_moves = []
        if last_move is None:
            # First move: any empty cell is valid
            for action in range(16):
                row = action // 4
                col = action % 4
                if self.board[row, col] == 0:
                    valid_moves.append(action)
        else:
            # Must move to an empty adjacent cell
            last_row, last_col = last_move
            adjacent_positions = []
            # Above
            if last_row > 0:
                adjacent_positions.append((last_row - 1, last_col))
            # Below
            if last_row < 3:
                adjacent_positions.append((last_row + 1, last_col))
            # Left
            if last_col > 0:
                adjacent_positions.append((last_row, last_col - 1))
            # Right
            if last_col < 3:
                adjacent_positions.append((last_row, last_col + 1))
            for pos in adjacent_positions:
                row, col = pos
                if self.board[row, col] == 0:
                    action = row * 4 + col
                    valid_moves.append(action)
        return valid_moves
