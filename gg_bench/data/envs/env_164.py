import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(25)  # 5x5 grid, actions 0-24
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        # Initialize the grid and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), -10, True, False, {}

        # Map action to grid coordinates
        row = action // 5  # Row index (0-4)
        col = action % 5  # Column index (0-4)

        # Check if the action is valid
        if self.grid[row, col] != 0:
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Place the player's marker on the grid
        self.grid[row, col] = self.current_player

        # Check for a win
        if self.check_win(self.current_player):
            self.done = True
            return self.grid.copy(), 1, True, False, {}

        # Check for a draw (should not happen in this game)
        if np.all(self.grid != 0):
            self.done = True
            return self.grid.copy(), 0, True, False, {}

        # Switch to the other player
        self.current_player *= -1

        return self.grid.copy(), 0, False, False, {}

    def check_win(self, player):
        visited = set()
        stack = []

        if player == 1:
            # Player 1: vertical path from bottom row (row 0) to top row (row 4)
            for col in range(5):
                if self.grid[0, col] == player:
                    stack.append((0, col))

            target_row = 4  # Top row index
            while stack:
                row, col = stack.pop()
                if (row, col) in visited:
                    continue
                visited.add((row, col))
                if row == target_row:
                    return True  # Reached the top row

                # Explore neighbors (up, down, left, right)
                neighbors = []
                if row > 0:
                    neighbors.append((row - 1, col))
                if row < 4:
                    neighbors.append((row + 1, col))
                if col > 0:
                    neighbors.append((row, col - 1))
                if col < 4:
                    neighbors.append((row, col + 1))
                for n_row, n_col in neighbors:
                    if (
                        self.grid[n_row, n_col] == player
                        and (n_row, n_col) not in visited
                    ):
                        stack.append((n_row, n_col))
        else:
            # Player -1: horizontal path from leftmost column (col 0) to rightmost column (col 4)
            for row in range(5):
                if self.grid[row, 0] == player:
                    stack.append((row, 0))

            target_col = 4  # Rightmost column index
            while stack:
                row, col = stack.pop()
                if (row, col) in visited:
                    continue
                visited.add((row, col))
                if col == target_col:
                    return True  # Reached the rightmost column

                # Explore neighbors (up, down, left, right)
                neighbors = []
                if row > 0:
                    neighbors.append((row - 1, col))
                if row < 4:
                    neighbors.append((row + 1, col))
                if col > 0:
                    neighbors.append((row, col - 1))
                if col < 4:
                    neighbors.append((row, col + 1))
                for n_row, n_col in neighbors:
                    if (
                        self.grid[n_row, n_col] == player
                        and (n_row, n_col) not in visited
                    ):
                        stack.append((n_row, n_col))
        return False

    def render(self):
        grid_str = "Current Grid:\n"
        for row in range(4, -1, -1):  # Print from top row to bottom row
            grid_str += f"{row + 1} ["
            for col in range(5):
                cell = self.grid[row, col]
                if cell == 1:
                    grid_str += "X"
                elif cell == -1:
                    grid_str += "O"
                else:
                    grid_str += " "
                if col != 4:
                    grid_str += "]["
                else:
                    grid_str += "]\n"
        grid_str += "    1  2  3  4  5\n"
        return grid_str

    def valid_moves(self):
        moves = []
        for action in range(25):
            row = action // 5
            col = action % 5
            if self.grid[row, col] == 0:
                moves.append(action)
        return moves
