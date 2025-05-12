import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 25 cells in the 5x5 grid
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(25,), dtype=np.int8)

        # Initialize the grid and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # The grid is a 1D array of size 25
        # 0: empty cell, 1: current player's symbol, -1: opponent's symbol
        self.grid = np.zeros(25, dtype=np.int8)
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if the action is within the valid range
        if action < 0 or action >= 25:
            return self._get_obs(), -10, True, False, {}

        # Check if the cell is empty
        if self.grid[action] != 0:
            # Invalid move: cell is already occupied
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Map action to row and column
        row = action // 5
        col = action % 5

        # Check if the move is valid (not adjacent to opponent's symbols)
        if not self._is_valid_move(row, col):
            # Invalid move: adjacent to opponent's symbol
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Place the current player's symbol on the grid
        self.grid[action] = 1  # Current player's symbol is 1

        # Check if the opponent has any valid moves
        if not self._opponent_has_valid_moves():
            # Opponent cannot move; current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch to the next player by inverting the grid and current player
        self.grid *= -1
        self.current_player *= -1

        return self._get_obs(), 0, False, False, {}

    def render(self):
        symbols = {0: "[ ]", 1: "[A]", -1: "[B]"}
        grid_str = ""
        for i in range(5):
            row_str = ""
            for j in range(5):
                idx = i * 5 + j
                row_str += symbols[self.grid[idx]]
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        valid_actions = []
        for idx in range(25):
            if self.grid[idx] == 0:
                row = idx // 5
                col = idx % 5
                if self._is_valid_move(row, col):
                    valid_actions.append(idx)
        return valid_actions

    def _get_obs(self):
        return self.grid.copy()

    def _is_valid_move(self, row, col):
        # Check if the cell is empty (already checked before calling this)
        # Check if the cell is adjacent to any opponent's symbol (-1)
        adjacent_cells = self._get_adjacent_cells(row, col)
        for r, c in adjacent_cells:
            idx = r * 5 + c
            if self.grid[idx] == -1:
                return False
        return True

    def _get_adjacent_cells(self, row, col):
        adjacent_cells = []
        for r in range(max(0, row - 1), min(4, row + 1) + 1):
            for c in range(max(0, col - 1), min(4, col + 1) + 1):
                if (r != row) or (c != col):
                    adjacent_cells.append((r, c))
        return adjacent_cells

    def _opponent_has_valid_moves(self):
        for idx in range(25):
            if self.grid[idx] == 0:
                row = idx // 5
                col = idx % 5
                if self._is_valid_move_for_opponent(row, col):
                    return True
        return False

    def _is_valid_move_for_opponent(self, row, col):
        # Check if the cell is adjacent to any of the current player's symbols (which are -1 from opponent's perspective)
        adjacent_cells = self._get_adjacent_cells(row, col)
        for r, c in adjacent_cells:
            idx = r * 5 + c
            if self.grid[idx] == -1:
                return False
        return True
