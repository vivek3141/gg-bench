import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 81 possible actions: 9 possible move positions * 9 possible block positions
        self.action_space = spaces.Discrete(81)
        # Observation is the state of the grid: 9 cells with values -1, 0, 1, or 2
        self.observation_space = spaces.Box(low=-1, high=2, shape=(9,), dtype=np.int8)

        # Initialize variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid
        # 0: empty, 1: Player 1, -1: Player 2, 2: Blocked
        self.grid = np.zeros(9, dtype=np.int8)
        self.grid[0] = 1  # Player 1 starts at position 0 (cell (0,0))
        self.grid[8] = -1  # Player 2 starts at position 8 (cell (2,2))
        self.current_player = 1
        self.done = False
        return self.grid.copy(), {}

    def step(self, action):
        if self.done:
            return self.grid.copy(), -10, True, False, {}

        move_pos = action // 9
        block_pos = action % 9

        # Check if move is valid
        if not self.is_valid_move(move_pos):
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Move the player
        current_pos = self.get_player_position(self.current_player)
        self.grid[current_pos] = 0
        self.grid[move_pos] = self.current_player

        # Check if block is valid
        if not self.is_valid_block(move_pos, block_pos):
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Block the cell
        self.grid[block_pos] = 2  # Blocked cell

        # Check if opponent has valid moves
        opponent = -self.current_player
        if not self.has_valid_moves(opponent):
            self.done = True
            return self.grid.copy(), 1, True, False, {}

        # Switch current player
        self.current_player *= -1
        return self.grid.copy(), 0, False, False, {}

    def render(self):
        symbols = {0: ".", 1: "P1", -1: "P2", 2: "X"}
        grid_str = ""
        for i in range(3):
            row = ""
            for j in range(3):
                val = self.grid[i * 3 + j]
                cell_str = f"{symbols[val]:>2}"
                row += cell_str + " "
            grid_str += row.strip() + "\n"
        return grid_str.strip()

    def valid_moves(self):
        valid_actions = []
        current_pos = self.get_player_position(self.current_player)
        adjacent_moves = self.get_adjacent_positions(current_pos, move=True)
        for move_pos in adjacent_moves:
            if self.grid[move_pos] == 0:
                adjacent_blocks = self.get_adjacent_positions(move_pos, move=False)
                for block_pos in adjacent_blocks:
                    if self.grid[block_pos] == 0:
                        action = move_pos * 9 + block_pos
                        valid_actions.append(action)
        return valid_actions

    def is_valid_move(self, move_pos):
        current_pos = self.get_player_position(self.current_player)
        if self.grid[move_pos] != 0:
            return False
        if move_pos not in self.get_adjacent_positions(current_pos, move=True):
            return False
        return True

    def is_valid_block(self, move_pos, block_pos):
        if self.grid[block_pos] != 0:
            return False
        if block_pos not in self.get_adjacent_positions(move_pos, move=False):
            return False
        return True

    def get_player_position(self, player):
        positions = np.where(self.grid == player)[0]
        return positions[0] if positions.size > 0 else None

    def get_adjacent_positions(self, pos, move):
        row = pos // 3
        col = pos % 3
        positions = []
        directions = (
            [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if move
            else [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        )
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                positions.append(new_row * 3 + new_col)
        return positions

    def has_valid_moves(self, player):
        current_pos = self.get_player_position(player)
        if current_pos is None:
            return False
        adjacent_moves = self.get_adjacent_positions(current_pos, move=True)
        for move_pos in adjacent_moves:
            if self.grid[move_pos] == 0:
                adjacent_blocks = self.get_adjacent_positions(move_pos, move=False)
                for block_pos in adjacent_blocks:
                    if self.grid[block_pos] == 0:
                        return True
        return False
