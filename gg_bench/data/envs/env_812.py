import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete(81) corresponding to (number_index * 9) + position_index
        self.action_space = spaces.Discrete(81)

        # Observation space: concatenation of grid, available numbers for both players, secret target, and current player
        self.observation_space = spaces.Box(
            low=np.array([-9] * 9 + [0] * 9 + [0] * 9 + [10] + [1], dtype=np.int32),
            high=np.array([9] * 9 + [1] * 9 + [1] * 9 + [20] + [2], dtype=np.int32),
            dtype=np.int32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid and player states
        self.grid = np.zeros(9, dtype=np.int32)
        self.available_numbers_p1 = np.ones(9, dtype=np.int32)
        self.available_numbers_p2 = np.ones(9, dtype=np.int32)
        self.secret_target = np.random.randint(10, 21)  # 10 to 20 inclusive
        self.current_player = 1
        self.done = False

        # Return the initial observation and info
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            # Game already finished
            return self._get_observation(), 0, True, False, {}

        number_index = action // 9  # From 0 to 8
        position_index = action % 9  # From 0 to 8
        number = number_index + 1  # Numbers 1 to 9

        # Check if the number is available and position is empty
        if self.current_player == 1:
            if (
                self.available_numbers_p1[number_index] == 0
                or self.grid[position_index] != 0
            ):
                # Invalid move
                self.done = True
                return self._get_observation(), -10, True, False, {}
        else:
            if (
                self.available_numbers_p2[number_index] == 0
                or self.grid[position_index] != 0
            ):
                # Invalid move
                self.done = True
                return self._get_observation(), -10, True, False, {}

        # Valid move: Place the number on the grid
        if self.current_player == 1:
            self.grid[position_index] = number
            self.available_numbers_p1[number_index] = 0
        else:
            self.grid[position_index] = -number
            self.available_numbers_p2[number_index] = 0

        # Check for victory
        win = self.check_victory()

        if win:
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Check for loss if the grid is full
        if np.all(self.grid != 0):
            # Current player loses
            self.done = True
            return self._get_observation(), -1, True, False, {}

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        # Return the observation and info
        return self._get_observation(), 0, False, False, {}

    def _get_observation(self):
        # Concatenate the grid, available numbers, secret target, and current player
        observation = np.concatenate(
            (
                self.grid,
                self.available_numbers_p1,
                self.available_numbers_p2,
                [self.secret_target],
                [self.current_player],
            )
        )
        return observation

    def check_victory(self):
        # Define winning combinations: rows, columns, and diagonals
        win_indices = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],  # Rows
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],  # Columns
            [0, 4, 8],
            [2, 4, 6],  # Diagonals
        ]

        for indices in win_indices:
            line = self.grid[indices]
            if self.current_player == 1:
                if np.all(line > 0):
                    # Sum the player's numbers
                    if np.sum(line) == self.secret_target:
                        return True
            else:
                if np.all(line < 0):
                    # Sum the player's numbers (absolute values)
                    if -np.sum(line) == self.secret_target:
                        return True
        return False

    def render(self):
        # Create a visual representation of the grid
        grid_str = "-------------\n"
        for i in range(3):
            grid_str += "|"
            for j in range(3):
                idx = i * 3 + j
                val = self.grid[idx]
                if val > 0:
                    grid_str += f" {val} |"  # Player 1's numbers
                elif val < 0:
                    grid_str += f"({-val})|"  # Player 2's numbers in parentheses
                else:
                    grid_str += "   |"
            grid_str += "\n-------------\n"
        return grid_str

    def valid_moves(self):
        # Generate a list of valid actions
        valid_actions = []
        for number_index in range(9):
            if self.current_player == 1:
                if self.available_numbers_p1[number_index] == 0:
                    continue
            else:
                if self.available_numbers_p2[number_index] == 0:
                    continue
            for position_index in range(9):
                if self.grid[position_index] == 0:
                    action = number_index * 9 + position_index
                    valid_actions.append(action)
        return valid_actions
