import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 12 possible actions:
        # Actions 0-5: Rows 0-2, Operations 0 (Add 1) or 1 (Subtract 1)
        # Actions 6-11: Columns 0-2, Operations 0 (Add 1) or 1 (Subtract 1)
        self.action_space = spaces.Discrete(12)

        # Observation space: 3x3 grid with integer values between -15 and 15
        self.observation_space = spaces.Box(
            low=-15, high=15, shape=(3, 3), dtype=np.int32
        )

        # Initialize the grid and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid with random integers between -5 and 5, excluding zero
        self.grid = self.np_random.integers(-5, 6, size=(3, 3))
        zero_positions = self.grid == 0
        while zero_positions.any():
            self.grid[zero_positions] = self.np_random.choice(
                [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], size=zero_positions.sum()
            )
            zero_positions = self.grid == 0

        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if action < 0 or action >= 12:
            raise ValueError("Invalid action.")

        if self.done:
            raise Exception("Game is over. Please reset the environment.")

        # Map action to line_type, line_number, operation
        # Actions 0-5: Rows
        # Actions 6-11: Columns
        # Operation 0: Add 1, Operation 1: Subtract 1

        if action < 6:
            line_type = "row"
            line_number = action // 2  # 0, 1, or 2
            operation = action % 2  # 0: Add 1, 1: Subtract 1
        else:
            line_type = "column"
            line_number = (action - 6) // 2  # 0, 1, or 2
            operation = (action - 6) % 2  # 0: Add 1, 1: Subtract 1

        # Apply the operation to the selected line
        if line_type == "row":
            if operation == 0:
                self.grid[line_number, :] += 1
            else:
                self.grid[line_number, :] -= 1
        else:
            if operation == 0:
                self.grid[:, line_number] += 1
            else:
                self.grid[:, line_number] -= 1

        # Check for win condition
        if np.all(self.grid == 0):
            self.done = True
            reward = 1  # Current player wins
        else:
            reward = -10  # Valid move made

        # Switch to the other player
        self.current_player = 3 - self.current_player  # Switches between 1 and 2

        return (
            self.grid.copy(),
            reward,
            self.done,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        # Return a visual representation of the grid as a string
        grid_str = "Row\\Col |  1   2   3\n"
        grid_str += "-----------------------\n"
        for i in range(3):
            row_str = f"   {i+1}    |"
            for j in range(3):
                val = self.grid[i, j]
                row_str += f" {val:>2} "
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        # All moves are always valid in this game
        return list(range(12))
