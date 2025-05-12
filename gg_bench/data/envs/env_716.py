import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The board is a 7x7 grid, so there are 49 possible actions (cells)
        self.action_space = spaces.Discrete(49)

        # The observation is the state of the board
        # Each cell can be -1 (Player O), 0 (empty), or 1 (Player X)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7, 7), dtype=np.int8)

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((7, 7), dtype=np.int8)
        self.current_player = 1  # Player 1 is 1, Player 2 is -1
        self.last_moves = {
            1: None,
            -1: None,
        }  # Keep track of last moves for each player
        self.move_number = 0  # Count the number of moves made
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}  # Game is over

        row = action // 7
        col = action % 7

        # Check if the action is within the board boundaries
        if not (0 <= row < 7 and 0 <= col < 7):
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Check if the cell is empty
        if self.board[row, col] != 0:
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Validate the move according to the game rules
        valid = False
        if self.move_number == 0 and self.current_player == 1:
            # First move by Player 1 can be any empty cell
            valid = True
        else:
            opponent = -self.current_player
            opp_last_move = self.last_moves[opponent]
            if opp_last_move is None:
                # Should not happen if game is progressing correctly
                return self.board.copy(), -10, True, False, {}  # Invalid state
            opp_row, opp_col = opp_last_move
            # Check same row or same column
            if row == opp_row or col == opp_col:
                valid = True
            # Check diagonals
            elif (row - col) == (opp_row - opp_col):
                valid = True
            elif (row + col) == (opp_row + opp_col):
                valid = True

        if not valid:
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Place the marker on the board
        self.board[row, col] = self.current_player
        self.last_moves[self.current_player] = (row, col)
        self.move_number += 1

        # Check if the opponent has any valid moves
        opponent = -self.current_player
        opp_valid_moves = self.get_valid_moves(opponent)
        if not opp_valid_moves:
            # Opponent has no valid moves; current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}  # Current player wins

        # Switch to the opponent
        self.current_player = opponent
        return self.board.copy(), 0, False, False, {}  # Continue game

    def render(self):
        board_str = "   A B C D E F G\n"
        for row in range(7):
            board_str += f" {row + 1} "
            for col in range(7):
                cell = self.board[row, col]
                if cell == 1:
                    board_str += "X "
                elif cell == -1:
                    board_str += "O "
                else:
                    board_str += ". "
            board_str += "\n"
        return board_str

    def valid_moves(self):
        return self.get_valid_moves(self.current_player)

    def get_valid_moves(self, player):
        valid_actions = []
        if self.done:
            return valid_actions  # No valid moves if the game is over

        if self.move_number == 0 and player == 1:
            # First move by Player 1 can be any empty cell
            empty_cells = np.argwhere(self.board == 0)
            for cell in empty_cells:
                row, col = cell
                action = row * 7 + col
                valid_actions.append(action)
            return valid_actions

        opponent = -player
        opp_last_move = self.last_moves[opponent]
        if opp_last_move is None:
            # No valid moves if opponent's last move is undefined
            return valid_actions

        opp_row, opp_col = opp_last_move

        # Generate all possible cells in the same row, column, and diagonals
        possible_cells = set()

        # Same row
        for col in range(7):
            possible_cells.add((opp_row, col))

        # Same column
        for row in range(7):
            possible_cells.add((row, opp_col))

        # Diagonals (row - col == constant)
        diag1_const = opp_row - opp_col
        for row in range(7):
            col = row - diag1_const
            if 0 <= col < 7:
                possible_cells.add((row, col))

        # Diagonals (row + col == constant)
        diag2_const = opp_row + opp_col
        for row in range(7):
            col = diag2_const - row
            if 0 <= col < 7:
                possible_cells.add((row, col))

        # Filter empty cells
        for cell in possible_cells:
            row, col = cell
            if self.board[row, col] == 0:
                action = row * 7 + col
                valid_actions.append(action)

        return valid_actions
