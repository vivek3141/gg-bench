import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: Discrete(25 * 25) = 625 possible actions
        self.action_space = spaces.Discrete(625)

        # Define observation space: 25 cells with values -1 (removed), 0 (empty), 1 (Player 1), 2 (Player 2)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(25,), dtype=np.int8)

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board
        self.board = np.zeros(25, dtype=np.int8)
        # Player 1 starts
        self.current_player = 1
        # No marker placed yet
        self.last_placed_cell = None
        # Game is not over
        self.done = False
        return self.board.copy(), {}  # Return observation and empty info

    def step(self, action):
        if self.done:
            # Game is over, invalid action
            return self.board.copy(), -10, True, False, {}

        # Decode the action into placement and blockade cells
        placement_cell = action // 25  # Integer division
        blockade_cell = action % 25  # Remainder

        # Validate the action
        if not self.is_valid_action(placement_cell, blockade_cell):
            # Invalid action
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Place the current player's marker
        self.board[placement_cell] = self.current_player
        self.last_placed_cell = placement_cell

        # Remove the blockade cell
        self.board[blockade_cell] = -1  # Mark as removed

        # Check if the opponent has valid moves
        opponent = 1 if self.current_player == 2 else 2
        if not self.has_valid_moves(opponent):
            # Opponent cannot move; current player wins
            self.done = True
            reward = 1  # Current player wins
        else:
            # Switch to the opponent
            self.current_player = opponent
            reward = 0  # No immediate reward

        return self.board.copy(), reward, self.done, False, {}

    def render(self):
        # Visual representation of the board
        board_str = ""
        for i in range(5):
            row_str = ""
            for j in range(5):
                idx = i * 5 + j
                cell_value = self.board[idx]
                if cell_value == 0:
                    row_str += " . "
                elif cell_value == 1:
                    row_str += " A "
                elif cell_value == 2:
                    row_str += " B "
                elif cell_value == -1:
                    row_str += " X "
            board_str += row_str + "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        # Get valid placement cells for the current player
        valid_placement_cells = self.get_valid_placement_cells(self.current_player)
        # Get valid blockade cells (empty cells)
        valid_blockade_cells = np.where(self.board == 0)[0].tolist()

        # Generate all valid action combinations
        for placement_cell in valid_placement_cells:
            for blockade_cell in valid_blockade_cells:
                if placement_cell != blockade_cell:
                    action = placement_cell * 25 + blockade_cell
                    valid_actions.append(action)
        return valid_actions

    def is_valid_action(self, placement_cell, blockade_cell):
        # Check if placement cell is empty
        if self.board[placement_cell] != 0:
            return False

        # On the first move, Player 1 can place on any empty cell
        if self.last_placed_cell is None:
            if self.current_player != 1:
                return False  # Shouldn't happen; Player 1 starts
        else:
            # Placement cell must be adjacent to the last placed marker
            if not self.are_adjacent(placement_cell, self.last_placed_cell):
                return False

        # Check if blockade cell is valid
        if self.board[blockade_cell] != 0:
            return False  # Cell is not empty
        if placement_cell == blockade_cell:
            return False  # Cannot blockade the cell just placed

        return True

    def are_adjacent(self, cell1, cell2):
        # Check if two cells are adjacent (up/down/left/right)
        row1, col1 = divmod(cell1, 5)
        row2, col2 = divmod(cell2, 5)
        return (abs(row1 - row2) == 1 and col1 == col2) or (
            abs(col1 - col2) == 1 and row1 == row2
        )

    def get_valid_placement_cells(self, player):
        if self.last_placed_cell is None:
            if player != 1:
                return []  # Only Player 1 can move first
            # All empty cells are valid for Player 1's first move
            return np.where(self.board == 0)[0].tolist()
        else:
            # Get adjacent empty cells to the last placed marker
            return self.get_adjacent_empty_cells(self.last_placed_cell)

    def get_adjacent_empty_cells(self, cell):
        adjacent_cells = []
        row, col = divmod(cell, 5)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                neighbor_cell = new_row * 5 + new_col
                if self.board[neighbor_cell] == 0:
                    adjacent_cells.append(neighbor_cell)
        return adjacent_cells

    def has_valid_moves(self, player):
        # Check if the player has any valid moves
        valid_placement_cells = self.get_valid_placement_cells(player)
        if not valid_placement_cells:
            # No valid placements available
            return False
        valid_blockade_cells = np.where(self.board == 0)[0]
        if len(valid_blockade_cells) == 0:
            # No empty cells to blockade
            return False
        return True
