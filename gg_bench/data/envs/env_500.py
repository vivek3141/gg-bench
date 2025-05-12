import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space (25 cells to claim)
        self.action_space = spaces.Discrete(25)

        # Define observation space
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the 5x5 grid with zeros
        self.board = np.zeros((5, 5), dtype=np.int8)
        # Player 1 uses 1 (X), Player 2 uses -1 (O)
        self.current_player = 1
        self.done = False
        return self.board.copy(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Convert action to 2D coordinates
        row, col = divmod(action, 5)

        # Check if the action is valid
        if self.board[row, col] != 0:
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Claim the cell
        self.board[row, col] = self.current_player

        # Capture logic
        self._capture_cells(row, col)

        # Check win condition
        player_cells = np.sum(self.board == self.current_player)
        if player_cells >= 13:
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch to the other player
        self.current_player *= -1

        return self.board.copy(), 0, False, False, {}

    def _capture_cells(self, row, col):
        # Check if newly claimed cell is adjacent to any of player's own cells (excluding itself)
        adjacent_own = self._has_adjacent_own_cells(row, col)

        if adjacent_own:
            # Get list of adjacent opponent's cells
            opponent_cells = self._get_adjacent_opponent_cells(row, col)
            if opponent_cells:
                # Capture one opponent's cell (choose the first in the list)
                opp_row, opp_col = opponent_cells[0]
                self.board[opp_row, opp_col] = self.current_player

    def _has_adjacent_own_cells(self, row, col):
        # Check adjacent cells for own symbols (excluding the newly claimed cell)
        own_symbol = self.current_player
        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < 5 and 0 <= n_col < 5:
                if self.board[n_row, n_col] == own_symbol:
                    return True
        return False

    def _get_adjacent_opponent_cells(self, row, col):
        # Get list of adjacent opponent's cells
        opponent_symbol = -self.current_player
        opponent_cells = []
        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < 5 and 0 <= n_col < 5:
                if self.board[n_row, n_col] == opponent_symbol:
                    opponent_cells.append((n_row, n_col))
        return opponent_cells

    def render(self):
        # Generate a string representation of the board
        symbol_map = {1: "X", -1: "O", 0: "."}
        board_str = ""
        for row in range(5):
            for col in range(5):
                board_str += symbol_map[self.board[row, col]] + " "
            board_str = board_str.strip() + "\n"
        return board_str.strip()

    def valid_moves(self):
        # Return a list of indices of unclaimed cells
        valid_moves = []
        for index in range(25):
            row, col = divmod(index, 5)
            if self.board[row, col] == 0:
                valid_moves.append(index)
        return valid_moves
