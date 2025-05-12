import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: 9 numbers * 9 positions = 81 possible actions
        self.action_space = spaces.Discrete(81)

        # Observation space: Player grids and available numbers
        self.observation_space = spaces.Box(low=0, high=9, shape=(27,), dtype=int)

        # Initialize environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize player grids (two players)
        self.player_grids = [np.zeros(9, dtype=int), np.zeros(9, dtype=int)]

        # All numbers from 1 to 9 are available
        self.available_numbers = np.ones(9, dtype=int)

        # Current player (0 or 1)
        self.current_player = 0

        # Game not done
        self.done = False

        # Step counter
        self.steps = 0

        # Maximum steps to prevent infinite loops
        self.max_steps = 100

        # Return initial observation and info
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Game is over. Please reset the environment.")

        # Decode action into number and cell index
        number_index = action // 9
        cell_index = action % 9
        number = number_index + 1  # Numbers range from 1 to 9

        # Check if the number is available
        if self.available_numbers[number_index] == 0:
            # Invalid move: number already used
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Check if the cell is empty in the current player's grid
        if self.player_grids[self.current_player][cell_index] != 0:
            # Invalid move: cell already occupied
            reward = -10
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Valid move: place the number on the grid
        self.player_grids[self.current_player][cell_index] = number
        self.available_numbers[number_index] = 0  # Mark number as used

        # Check for win condition
        if self._check_win(self.player_grids[self.current_player]):
            reward = 1
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Increment step counter and check for maximum steps
        self.steps += 1
        if self.steps >= self.max_steps:
            # Game ends without a winner (in practice)
            reward = 0
            self.done = True
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Switch to the other player
        self.current_player = 1 - self.current_player

        # Continue game
        reward = 0
        observation = self._get_observation()
        return observation, reward, False, False, {}

    def render(self):
        # Generate string representation of the game state
        grid_string = ""
        for player in range(2):
            grid_string += f"Player {player + 1}'s Grid:\n"
            grid = self.player_grids[player]
            for i in range(3):
                row_str = "|"
                for j in range(3):
                    value = grid[i * 3 + j]
                    if value == 0:
                        row_str += "   |"
                    else:
                        row_str += f" {value} |"
                grid_string += row_str + "\n"
            grid_string += "-------------\n"
        grid_string += "Available Numbers:\n"
        for i in range(9):
            if self.available_numbers[i]:
                grid_string += f"{i + 1} "
            else:
                grid_string += "   "
        grid_string += "\n"
        return grid_string

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for number_index in range(9):
            if self.available_numbers[number_index]:
                for cell_index in range(9):
                    if self.player_grids[self.current_player][cell_index] == 0:
                        action = number_index * 9 + cell_index
                        valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Combine player grids and available numbers into a single observation
        own_grid = self.player_grids[self.current_player]
        opponent_grid = self.player_grids[1 - self.current_player]
        available_numbers = self.available_numbers
        observation = np.concatenate([own_grid, opponent_grid, available_numbers])
        return observation

    def _check_win(self, grid):
        # Check all rows, columns, and diagonals for a sum of 15
        win_indices = [
            [0, 1, 2],  # Row 1
            [3, 4, 5],  # Row 2
            [6, 7, 8],  # Row 3
            [0, 3, 6],  # Column 1
            [1, 4, 7],  # Column 2
            [2, 5, 8],  # Column 3
            [0, 4, 8],  # Diagonal 1
            [2, 4, 6],  # Diagonal 2
        ]
        for indices in win_indices:
            values = [grid[i] for i in indices]
            if 0 not in values:
                if sum(values) == 15:
                    return True
        return False
