import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=np.int8)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid
        self.grid = np.zeros((5, 5), dtype=np.int8)
        # Place obstacles
        obstacle_positions = [(2, 0), (0, 2), (2, 2), (4, 2), (2, 4)]
        for row, col in obstacle_positions:
            self.grid[row, col] = -1  # Obstacles represented by -1
        # Place players
        self.player_positions = {
            1: (0, 0),  # Player 1 starts at A1
            2: (4, 4),  # Player 2 starts at E5
        }
        self.grid[0, 0] = 1  # Player 1 represented by 1
        self.grid[4, 4] = 2  # Player 2 represented by 2
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}
        # Map action to movement
        move_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }
        dr, dc = move_map[action]
        row, col = self.player_positions[self.current_player]
        new_row, new_col = row + dr, col + dc
        # Check grid boundaries
        if not (0 <= new_row < 5 and 0 <= new_col < 5):
            self.done = True
            return self.grid.copy(), -10, True, False, {}
        target_cell = self.grid[new_row, new_col]
        # Check if move is into an obstacle or occupied cell
        if target_cell != 0:
            self.done = True
            return self.grid.copy(), -10, True, False, {}
        # Valid move; update the grid and player's position
        self.grid[row, col] = 0
        self.grid[new_row, new_col] = self.current_player
        self.player_positions[self.current_player] = (new_row, new_col)
        # Check for win condition
        opponent_start_positions = {
            1: (4, 4),  # Player 1 wins by reaching E5
            2: (0, 0),  # Player 2 wins by reaching A1
        }
        if (new_row, new_col) == opponent_start_positions[self.current_player]:
            self.done = True
            return self.grid.copy(), 1, True, False, {}
        # Switch to the other player
        self.current_player = 1 if self.current_player == 2 else 2
        return self.grid.copy(), 0, False, False, {}

    def render(self):
        grid_str = "    A   B   C   D   E\n"
        grid_str += "  +" + "---+" * 5 + "\n"
        for row_idx in range(5):
            row_label = str(row_idx + 1)
            grid_str += row_label + " |"
            for col_idx in range(5):
                cell_value = self.grid[row_idx, col_idx]
                if cell_value == 0:
                    grid_str += "   |"
                elif cell_value == -1:
                    grid_str += " X |"
                elif cell_value == 1:
                    grid_str += "P1 |"
                elif cell_value == 2:
                    grid_str += "P2 |"
            grid_str += "\n  +" + "---+" * 5 + "\n"
        return grid_str

    def valid_moves(self):
        valid_actions = []
        move_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }
        row, col = self.player_positions[self.current_player]
        for action, (dr, dc) in move_map.items():
            new_row, new_col = row + dr, col + dc
            # Check grid boundaries
            if not (0 <= new_row < 5 and 0 <= new_col < 5):
                continue
            target_cell = self.grid[new_row, new_col]
            # Check if move is into an empty cell
            if target_cell == 0:
                valid_actions.append(action)
        return valid_actions
