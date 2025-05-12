import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The grid is 5x5, so there are 25 possible actions
        self.action_space = spaces.Discrete(25)

        # Observation space is a 5x5 grid with values:
        # 0: empty cell
        # 1: Player 1's marker (X)
        # 2: Player 2's marker (O)
        # 3: Blocked cell (#)
        self.observation_space = spaces.Box(low=0, high=3, shape=(5, 5), dtype=np.int8)

        # Initialize the board and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid to empty
        self.grid = np.zeros((5, 5), dtype=np.int8)

        # Decide who starts first (Player 1: 1 or Player 2: 2)
        self.current_player = 1

        # Internal flag for game over
        self.done = False

        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        row = action // 5
        col = action % 5

        # Check if action is valid
        if action < 0 or action >= 25 or self.grid[row, col] != 0:
            # Invalid action
            self.done = True
            return (
                self.grid.copy(),
                -10,
                True,
                False,
                {},
            )

        # Place the player's marker
        self.grid[row, col] = self.current_player

        # Block orthogonally adjacent cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 5 and 0 <= c < 5:
                if self.grid[r, c] == 0:
                    self.grid[r, c] = 3  # Mark as blocked

        # Switch player
        previous_player = self.current_player
        self.current_player = 1 if self.current_player == 2 else 2

        # Check if the new current player has any valid moves
        valid_moves = self.valid_moves()

        if not valid_moves:
            # The current player has no valid moves; previous player wins
            self.done = True
            reward = 1  # Previous player wins
            return self.grid.copy(), reward, True, False, {}

        # Continue the game
        return self.grid.copy(), 0, False, False, {}

    def render(self):
        # Create a string representation of the grid
        symbols = {0: "   ", 1: " X ", 2: " O ", 3: " # "}
        grid_str = "    1   2   3   4   5\n"
        grid_str += "  +" + "---+" * 5 + "\n"
        for i in range(5):
            row_str = f"{i+1} |"
            for j in range(5):
                cell_value = self.grid[i, j]
                row_str += symbols[cell_value] + "|"
            grid_str += row_str + "\n"
            grid_str += "  +" + "---+" * 5 + "\n"
        return grid_str

    def valid_moves(self):
        # Return the list of valid move indices for the current player
        valid_moves = []
        for action in range(25):
            row = action // 5
            col = action % 5
            if self.grid[row, col] == 0:
                valid_moves.append(action)
        return valid_moves
