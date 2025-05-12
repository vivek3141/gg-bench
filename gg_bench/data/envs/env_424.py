import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.grid_size = 5
        self.action_space = spaces.Discrete(
            self.grid_size**2
        )  # 25 possible cells (0 to 24)
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(self.grid_size**2,), dtype=np.int8
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid
        self.grid = np.zeros(self.grid_size**2, dtype=np.int8)
        # Place Trail Blazer A at (0,0)
        self.grid[0] = 1  # 1 represents Trail Blazer A
        self.pos_A = (0, 0)
        # Place Trail Blazer B at (4,4)
        self.grid[self.grid_size**2 - 1] = 2  # 2 represents Trail Blazer B
        self.pos_B = (self.grid_size - 1, self.grid_size - 1)
        self.current_player = 1  # 1 for Player A, 2 for Player B
        self.done = False
        self.info = {}
        return self.grid.copy(), self.info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, self.info
        # Get current player's position
        if self.current_player == 1:
            curr_pos = self.pos_A
            opponent_base = (self.grid_size - 1, self.grid_size - 1)
        else:
            curr_pos = self.pos_B
            opponent_base = (0, 0)
        # Convert action (0-24) to position (row, col)
        dest_row = action // self.grid_size
        dest_col = action % self.grid_size
        dest_pos = (dest_row, dest_col)
        # Check if destination is adjacent
        row_diff = abs(dest_row - curr_pos[0])
        col_diff = abs(dest_col - curr_pos[1])
        if max(row_diff, col_diff) > 1 or (row_diff == 0 and col_diff == 0):
            # Invalid move (non-adjacent or same cell)
            self.done = True
            return self.grid.copy(), -10, True, False, self.info
        # Check if destination cell is unoccupied (0)
        dest_idx = dest_row * self.grid_size + dest_col
        if self.grid[dest_idx] != 0:
            # Invalid move (cell is occupied)
            self.done = True
            return self.grid.copy(), -10, True, False, self.info
        # Move the Trail Blazer
        # Leave trail on previous cell
        curr_idx = curr_pos[0] * self.grid_size + curr_pos[1]
        if self.current_player == 1:
            self.grid[curr_idx] = 3  # 3 represents trail of A
            self.grid[dest_idx] = 1  # Move Trail Blazer A
            self.pos_A = dest_pos
        else:
            self.grid[curr_idx] = 4  # 4 represents trail of B
            self.grid[dest_idx] = 2  # Move Trail Blazer B
            self.pos_B = dest_pos
        # Check for victory
        if dest_pos == opponent_base:
            # Current player wins by reaching opponent's base
            self.done = True
            return self.grid.copy(), 1, True, False, self.info
        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1
        # Check if next player has any valid moves
        if not self.has_valid_moves():
            # Next player has no valid moves, current player wins
            self.done = True
            return self.grid.copy(), 1, True, False, self.info
        return self.grid.copy(), 0, False, False, self.info

    def has_valid_moves(self):
        # Check if the current player has any valid moves
        if self.current_player == 1:
            curr_pos = self.pos_A
        else:
            curr_pos = self.pos_B
        valid_moves = self.get_valid_moves(curr_pos)
        return len(valid_moves) > 0

    def get_valid_moves(self, position):
        row, col = position
        valid_moves = []
        for d_row in [-1, 0, 1]:
            for d_col in [-1, 0, 1]:
                if d_row == 0 and d_col == 0:
                    continue
                new_row = row + d_row
                new_col = col + d_col
                if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                    idx = new_row * self.grid_size + new_col
                    if self.grid[idx] == 0:
                        valid_moves.append(idx)
        return valid_moves

    def valid_moves(self):
        # Return a list of valid actions (cell indices) for the current player
        if self.current_player == 1:
            curr_pos = self.pos_A
        else:
            curr_pos = self.pos_B
        return self.get_valid_moves(curr_pos)

    def render(self):
        # Return a string representation of the grid
        grid_str = ""
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                val = self.grid[idx]
                if val == 0:
                    cell = "."
                elif val == 1:
                    cell = "A"
                elif val == 2:
                    cell = "B"
                elif val == 3:
                    cell = "a"
                elif val == 4:
                    cell = "b"
                else:
                    cell = "?"
                row_str += cell + " "
            grid_str += row_str.strip() + "\n"
        return grid_str.strip()
