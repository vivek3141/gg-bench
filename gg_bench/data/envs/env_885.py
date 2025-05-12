import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(625) for move and blockade placement combinations
        self.action_space = spaces.Discrete(625)

        # Observation space: 5x5 grid with values:
        # 0: Empty, 1: Player A, 2: Player B, 3: Blockade
        self.observation_space = spaces.Box(low=0, high=3, shape=(5, 5), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.grid[0, 0] = 1  # Player A
        self.grid[4, 4] = 2  # Player B
        self.player_positions = {1: (0, 0), 2: (4, 4)}
        self.current_player = 1  # Player A starts
        self.opponent_player = 2
        self.terminated = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.terminated:
            return self.grid.copy(), -10, True, False, {}  # Already terminated

        # Decode action into move and blockade placement
        move_index = action // 25
        blockade_index = action % 25
        move_row = move_index // 5
        move_col = move_index % 5
        blockade_row = blockade_index // 5
        blockade_col = blockade_index % 5

        # Get current position
        cur_row, cur_col = self.player_positions[self.current_player]

        # Validate move: Must be adjacent (up, down, left, right)
        if not self.is_adjacent(cur_row, cur_col, move_row, move_col):
            self.terminated = True
            return self.grid.copy(), -10, True, False, {}  # Invalid move

        # Check move destination
        if not self.is_within_grid(move_row, move_col):
            self.terminated = True
            return self.grid.copy(), -10, True, False, {}  # Invalid move

        if self.grid[move_row, move_col] != 0:
            self.terminated = True
            return self.grid.copy(), -10, True, False, {}  # Invalid move

        # Move player
        self.grid[cur_row, cur_col] = 0
        self.grid[move_row, move_col] = self.current_player
        self.player_positions[self.current_player] = (move_row, move_col)

        # Check for win by reaching opponent's starting position
        opponent_start = (0, 0) if self.current_player == 2 else (4, 4)
        if (move_row, move_col) == opponent_start:
            self.terminated = True
            return self.grid.copy(), 1, True, False, {}  # Win

        # Validate blockade placement: Must be adjacent (including diagonals) to new position
        if not self.is_adjacent_diagonal(
            move_row, move_col, blockade_row, blockade_col
        ):
            self.terminated = True
            return self.grid.copy(), -10, True, False, {}  # Invalid blockade

        if not self.is_within_grid(blockade_row, blockade_col):
            self.terminated = True
            return self.grid.copy(), -10, True, False, {}  # Invalid blockade

        if self.grid[blockade_row, blockade_col] != 0:
            self.terminated = True
            return self.grid.copy(), -10, True, False, {}  # Invalid blockade

        # Place blockade
        self.grid[blockade_row, blockade_col] = 3  # Blockade

        # Check if opponent has any valid moves
        if not self.has_valid_moves(self.opponent_player):
            self.terminated = True
            return self.grid.copy(), 1, True, False, {}  # Win by blocking opponent

        # Switch players
        self.current_player, self.opponent_player = (
            self.opponent_player,
            self.current_player,
        )

        # As per the prompt, reward is -10 if the current player has played a valid move
        return self.grid.copy(), -10, False, False, {}  # Continue game

    def render(self):
        # Return a visual representation of the grid as a string
        grid_str = ""
        for i in range(5):
            row_str = ""
            for j in range(5):
                if self.grid[i, j] == 0:
                    row_str += ". "
                elif self.grid[i, j] == 1:
                    row_str += "A "
                elif self.grid[i, j] == 2:
                    row_str += "B "
                elif self.grid[i, j] == 3:
                    row_str += "X "
            grid_str += row_str.strip() + "\n"
        return grid_str.strip()

    def valid_moves(self):
        valid_actions = []
        cur_row, cur_col = self.player_positions[self.current_player]
        # Get possible moves
        moves = self.get_adjacent_positions(cur_row, cur_col)
        for move_row, move_col in moves:
            if (
                self.is_within_grid(move_row, move_col)
                and self.grid[move_row, move_col] == 0
            ):
                # Get valid blockade placements after moving
                blockades = self.get_adjacent_positions_diagonal(move_row, move_col)
                for blockade_row, blockade_col in blockades:
                    if (
                        self.is_within_grid(blockade_row, blockade_col)
                        and self.grid[blockade_row, blockade_col] == 0
                    ):
                        # Encode action
                        move_index = move_row * 5 + move_col
                        blockade_index = blockade_row * 5 + blockade_col
                        action = move_index * 25 + blockade_index
                        valid_actions.append(action)
        return valid_actions

    def is_adjacent(self, row1, col1, row2, col2):
        return abs(row1 - row2) + abs(col1 - col2) == 1

    def is_adjacent_diagonal(self, row1, col1, row2, col2):
        return max(abs(row1 - row2), abs(col1 - col2)) == 1

    def is_within_grid(self, row, col):
        return 0 <= row < 5 and 0 <= col < 5

    def has_valid_moves(self, player):
        row, col = self.player_positions[player]
        moves = self.get_adjacent_positions(row, col)
        for move_row, move_col in moves:
            if (
                self.is_within_grid(move_row, move_col)
                and self.grid[move_row, move_col] == 0
            ):
                return True
        return False

    def get_adjacent_positions(self, row, col):
        positions = [
            (row - 1, col),  # Up
            (row + 1, col),  # Down
            (row, col - 1),  # Left
            (row, col + 1),  # Right
        ]
        return positions

    def get_adjacent_positions_diagonal(self, row, col):
        positions = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                positions.append((row + dr, col + dc))
        return positions
