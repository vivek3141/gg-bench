import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)

        # Observation is a 3x3 grid with values:
        # 0: Empty
        # 1: Current player's position
        # -1: Opponent's position
        # -2: Blocked cell
        self.observation_space = spaces.Box(low=-2, high=1, shape=(3, 3), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid
        self.grid = np.zeros((3, 3), dtype=np.int8)

        # Set starting positions
        self.grid[0, 0] = 1  # Player A
        self.grid[2, 2] = -1  # Player B
        self.player_positions = {1: (0, 0), -1: (2, 2)}

        self.current_player = 1  # Start with Player A
        self.done = False

        return self.grid.copy(), {}

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, self.done, False, {}

        row = action // 3
        col = action % 3

        # Get current player's position
        pos_row, pos_col = self.player_positions[self.current_player]

        # Check if the move is valid
        if not self._is_valid_move(pos_row, pos_col, row, col):
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Move the player and block the previous cell
        self.grid[pos_row, pos_col] = -2  # Block the previous cell
        self.grid[row, col] = self.current_player
        self.player_positions[self.current_player] = (row, col)

        # Check if the opponent has any valid moves
        opponent = -self.current_player
        opp_row, opp_col = self.player_positions[opponent]
        if not self._has_valid_moves(opp_row, opp_col):
            self.done = True
            return self.grid.copy(), 1, True, False, {}

        # Switch to the opponent
        self.current_player = opponent
        return self.grid.copy(), 0, False, False, {}

    def render(self):
        grid_str = ""
        symbols = {0: " ", 1: "A", -1: "B", -2: "X"}
        for i in range(3):
            row_str = ""
            for j in range(3):
                cell = self.grid[i, j]
                row_str += f" {symbols[cell]} "
                if j < 2:
                    row_str += "|"
            grid_str += row_str
            if i < 2:
                grid_str += "\n-----------\n"
        return grid_str

    def valid_moves(self):
        pos_row, pos_col = self.player_positions[self.current_player]
        return self._get_valid_moves(pos_row, pos_col)

    def _is_valid_move(self, pos_row, pos_col, row, col):
        # Check adjacency
        if abs(row - pos_row) + abs(col - pos_col) != 1:
            return False
        # Check boundaries and cell status
        if not (0 <= row < 3 and 0 <= col < 3):
            return False
        if self.grid[row, col] != 0:
            return False
        return True

    def _get_valid_moves(self, row, col):
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for d in directions:
            new_row = row + d[0]
            new_col = col + d[1]
            if (
                0 <= new_row < 3
                and 0 <= new_col < 3
                and self.grid[new_row, new_col] == 0
            ):
                moves.append(new_row * 3 + new_col)
        return moves

    def _has_valid_moves(self, row, col):
        return bool(self._get_valid_moves(row, col))
