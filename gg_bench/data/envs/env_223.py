import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(25) for the 5x5 grid positions
        self.action_space = spaces.Discrete(25)
        # The observation space is a 5x5 grid with values:
        # 0 (empty), 1 (Player 1's marker), -1 (Player 2's marker), 2 (blocked)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # Player 1 starts the game
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def valid_moves(self):
        # Return a list of valid action indices (0-24) where the grid cell is empty
        return [idx for idx in range(25) if self.grid[idx // 5, idx % 5] == 0]

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        row = action // 5
        col = action % 5

        # Check if the move is valid (cell is empty and not blocked)
        if self.grid[row, col] != 0:
            # Invalid move
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Place the current player's marker
        self.grid[row, col] = self.current_player

        # Block the adjacent cells
        self.block_cells(row, col)

        # Switch to the other player
        self.current_player *= -1

        # Check if the opponent has any valid moves
        if not self.has_valid_moves():
            # Current player wins because opponent cannot move
            self.done = True
            return self.grid.copy(), 1, True, False, {}

        # Game continues
        return self.grid.copy(), -10, False, False, {}

    def block_cells(self, row, col):
        # Directions: Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Block adjacent cells if they are empty
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 5 and 0 <= c < 5 and self.grid[r, c] == 0:
                self.grid[r, c] = 2  # Mark as blocked

    def has_valid_moves(self):
        # Check if the current player has any valid moves
        return any(self.grid[idx // 5, idx % 5] == 0 for idx in range(25))

    def render(self):
        # Generate a string representation of the grid
        grid_str = "   1 2 3 4 5\n"
        grid_str += "  -------------\n"
        for i in range(5):
            row_str = f"{i+1} | "
            for j in range(5):
                val = self.grid[i, j]
                if val == 1:
                    row_str += "X "
                elif val == -1:
                    row_str += "O "
                elif val == 2:
                    row_str += "# "
                else:
                    row_str += "_ "
            grid_str += row_str + "\n"
        return grid_str
