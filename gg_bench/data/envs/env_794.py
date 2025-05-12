import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.grid_size = 5
        self.action_space = spaces.Discrete(self.grid_size**2)
        # Observation is the observed grid: -1 (unrevealed), 1-5 (clue numbers), or 10 (treasure)
        self.observation_space = spaces.Box(
            low=-1, high=10, shape=(self.grid_size, self.grid_size), dtype=np.int32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.current_player = 1  # Player 1 starts
        # Initialize hidden grid
        self.hidden_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        # Place treasure randomly
        treasure_row = self.np_random.integers(0, self.grid_size)
        treasure_col = self.np_random.integers(0, self.grid_size)
        self.treasure_location = (treasure_row, treasure_col)
        # Set treasure in hidden grid, represented by 0
        self.hidden_grid[treasure_row, treasure_col] = 0  # Treasure
        # Fill the rest of the grid with clue numbers
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) != self.treasure_location:
                    manhattan_distance = abs(row - treasure_row) + abs(
                        col - treasure_col
                    )
                    clue_number = min(
                        manhattan_distance, 5
                    )  # Clues are between 1 and 5
                    self.hidden_grid[row, col] = clue_number
        # Initialize observed grid
        self.observed_grid = -1 * np.ones(
            (self.grid_size, self.grid_size), dtype=np.int32
        )
        observation = np.copy(self.observed_grid)
        return observation, {}

    def step(self, action):
        if self.done:
            raise Exception("Game is already over. Please reset the environment.")
        # Convert action to (row, col)
        row = action // self.grid_size
        col = action % self.grid_size
        # Check if action is valid (within grid bounds)
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            reward = -10
            self.done = True
            info = {}
            return np.copy(self.observed_grid), reward, self.done, False, info
        # Check if cell has already been revealed
        if self.observed_grid[row, col] != -1:
            reward = -10
            self.done = True
            info = {}
            return np.copy(self.observed_grid), reward, self.done, False, info
        # Check hidden grid
        cell_value = self.hidden_grid[row, col]
        if cell_value == 0:
            # Found treasure
            self.observed_grid[row, col] = 10  # Represent treasure with 10
            reward = 1
            self.done = True
            info = {}
            return np.copy(self.observed_grid), reward, self.done, False, info
        else:
            # Reveal clue number
            self.observed_grid[row, col] = cell_value
            reward = 0
            # Check if only the treasure remains unrevealed
            if np.count_nonzero(self.observed_grid == -1) == 1:
                # Next player will select the remaining cell, which is the treasure
                pass  # Continue the game
            # Switch to next player
            self.current_player = (
                3 - self.current_player
            )  # Switch between Player 1 and 2
            self.done = False
            info = {}
            return np.copy(self.observed_grid), reward, self.done, False, info

    def render(self):
        # Return a string representation of the observed grid
        grid_str = ""
        for row in range(self.grid_size):
            row_str = ""
            for col in range(self.grid_size):
                val = self.observed_grid[row, col]
                if val == -1:
                    row_str += " . "
                elif val == 10:
                    row_str += " T "
                else:
                    row_str += f" {val} "
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        # Return list of action indices corresponding to unrevealed cells
        valid_actions = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.observed_grid[row, col] == -1:
                    action = row * self.grid_size + col
                    valid_actions.append(action)
        return valid_actions
