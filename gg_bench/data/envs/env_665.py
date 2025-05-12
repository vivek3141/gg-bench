import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 25 possible actions (placing on a 5x5 grid)
        self.action_space = spaces.Discrete(25)

        # Observation space is a 5x5 grid with values in {-1, 0, 1, 2}
        # -1: Blocked cell, 0: Empty cell, 1: Player 1's marker, 2: Player 2's marker
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)  # Empty grid
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        # Check if action is valid
        if not self.action_space.contains(action):
            # Invalid action index
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        row = action // 5
        col = action % 5

        cell_value = self.grid[row][col]

        if cell_value != 0:
            # Occupied or blocked cell
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Place the marker
        self.grid[row][col] = self.current_player

        # Blocking empty cells in the same row
        for c in range(5):
            if self.grid[row][c] == 0:
                self.grid[row][c] = -1  # Blocked cell

        # Blocking empty cells in the same column
        for r in range(5):
            if self.grid[r][col] == 0:
                self.grid[r][col] = -1  # Blocked cell

        # Save previous player for reward assignment
        previous_player = self.current_player

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if opponent (now current player) has any valid moves
        if not self.valid_moves():
            # Opponent cannot move; previous player wins
            self.done = True
            return self.grid.copy(), 1, True, False, {}

        # Game continues
        return self.grid.copy(), 0, False, False, {}

    def render(self):
        grid_str = "  0 1 2 3 4\n"
        grid_str += " -----------\n"
        for i in range(5):
            row_str = f"{i}| "
            for j in range(5):
                cell = self.grid[i][j]
                if cell == 0:
                    row_str += ". "
                elif cell == -1:
                    row_str += "# "
                elif cell == 1:
                    row_str += "X "
                elif cell == 2:
                    row_str += "O "
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        # Return a list of valid actions (empty and unblocked cells)
        valid_actions = []
        for i in range(25):
            row = i // 5
            col = i % 5
            if self.grid[row][col] == 0:
                valid_actions.append(i)
        return valid_actions
