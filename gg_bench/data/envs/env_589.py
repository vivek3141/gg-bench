import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, grid_size=5):
        super(CustomEnv, self).__init__()

        self.grid_size = grid_size
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.current_player = 1  # 1 for Player 1 (X), -1 for Player 2 (O)
        self.done = False
        return self.grid.copy(), {}

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        row = action // self.grid_size
        col = action % self.grid_size

        # Check for valid move
        if self.grid[row, col] != 0:
            self.done = True
            reward = -10
            return self.grid.copy(), reward, True, False, {}

        # Make move
        self.grid[row, col] = self.current_player

        # Check for win
        if self.check_win(self.current_player):
            self.done = True
            reward = 1
            return self.grid.copy(), reward, True, False, {}
        else:
            # Switch player
            self.current_player *= -1
            reward = 0
            return self.grid.copy(), reward, False, False, {}

    def render(self):
        # Return a string representation of the grid
        grid_str = ""
        for row in range(self.grid_size):
            row_str = ""
            for col in range(self.grid_size):
                val = self.grid[row, col]
                if val == 1:
                    row_str += "X "
                elif val == -1:
                    row_str += "O "
                else:
                    row_str += ". "
            grid_str += row_str.strip() + "\n"
        return grid_str.strip()

    def valid_moves(self):
        return [
            i for i in range(self.grid_size * self.grid_size) if self.grid.flat[i] == 0
        ]

    def check_win(self, player):
        """Check if the player has a winning path"""
        visited = np.zeros_like(self.grid, dtype=bool)
        if player == 1:
            # Player 1 aims to connect left to right
            for row in range(self.grid_size):
                if self.grid[row, 0] == player:
                    if self.dfs(player, row, 0, visited):
                        return True
        else:
            # Player 2 aims to connect top to bottom
            for col in range(self.grid_size):
                if self.grid[0, col] == player:
                    if self.dfs(player, 0, col, visited):
                        return True
        return False

    def dfs(self, player, row, col, visited):
        if visited[row, col]:
            return False
        visited[row, col] = True

        if player == 1 and col == self.grid_size - 1:
            return True
        if player == -1 and row == self.grid_size - 1:
            return True

        # Explore neighbors (including diagonals)
        for dr, dc in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]:
            r, c = row + dr, col + dc
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                if self.grid[r, c] == player and not visited[r, c]:
                    if self.dfs(player, r, c, visited):
                        return True
        return False
