import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.grid_size = 5

        # Define action and observation space
        # Actions: 0 - up, 1 - down, 2 - left, 3 - right
        self.action_space = spaces.Discrete(4)

        # Observation space: 5x5 grid with values:
        # 0 - Empty, -1 - Blocked, 1 - Player 1, 2 - Player 2
        self.observation_space = spaces.Box(
            low=-1, high=2, shape=(self.grid_size, self.grid_size), dtype=np.int8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Starting positions
        # Player 1 starts at (2, 0)
        # Player 2 starts at (2, 4)
        self.player_positions = {
            1: (2, 0),  # Row 3, Column 1 in 1-based indexing
            2: (2, 4),  # Row 3, Column 5 in 1-based indexing
        }

        self.grid[self.player_positions[1]] = 1
        self.grid[self.player_positions[2]] = 2

        self.current_player = 1
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}  # Game is over

        # Map action to movement
        action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        # Get the current player's position
        row, col = self.player_positions[self.current_player]
        # Get the movement delta
        d_row, d_col = action_map.get(action, (0, 0))
        new_row = row + d_row
        new_col = col + d_col

        # Check if the move is valid
        if (0 <= new_row < self.grid_size) and (0 <= new_col < self.grid_size):
            dest_cell = self.grid[new_row, new_col]
            if dest_cell == 0:
                # Valid move
                # Block the cell we're leaving
                self.grid[row, col] = -1  # Mark as blocked

                # Move to the new cell
                self.grid[new_row, new_col] = self.current_player
                self.player_positions[self.current_player] = (new_row, new_col)
            else:
                # Invalid move (cell is occupied or blocked)
                self.done = True
                return self.grid.copy(), -10, True, False, {}
        else:
            # Invalid move (out of bounds)
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if the new current player has any valid moves
        if not self.valid_moves():
            # Previous player wins
            self.done = True
            # Switch back to previous player to assign the reward correctly
            self.current_player = 2 if self.current_player == 1 else 1
            return self.grid.copy(), 1, True, False, {}

        # Game continues
        return self.grid.copy(), 0, False, False, {}

    def render(self):
        # Return a string representation of the grid
        display_grid = ""
        for row in self.grid:
            display_row = ""
            for cell in row:
                if cell == 0:
                    display_row += ". "
                elif cell == -1:
                    display_row += "# "
                elif cell == 1:
                    display_row += "1 "
                elif cell == 2:
                    display_row += "2 "
            display_grid += display_row.strip() + "\n"
        return display_grid.strip()

    def valid_moves(self):
        # Return a list of valid actions for the current player
        valid_actions = []
        action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        row, col = self.player_positions[self.current_player]

        for action, (d_row, d_col) in action_map.items():
            new_row = row + d_row
            new_col = col + d_col

            if (0 <= new_row < self.grid_size) and (0 <= new_col < self.grid_size):
                dest_cell = self.grid[new_row, new_col]
                if dest_cell == 0:
                    valid_actions.append(action)

        return valid_actions
