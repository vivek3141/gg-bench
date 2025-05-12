import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 5 possible moves
        self.action_space = spaces.Discrete(5)
        # Observation space: 5x5 grid with values -1, 0, 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        # Initialize the environment state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.grid[0, 2] = 1  # Player 1 starts at (0, 2)
        self.grid[4, 2] = -1  # Player 2 starts at (4, 2)
        self.player_positions = {1: (0, 2), -1: (4, 2)}
        self.current_player = 1
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self.grid.copy(), -10, True, False, {}  # Invalid move

        # Get current position
        row, col = self.player_positions[self.current_player]
        # Get movement for action
        new_row, new_col = self.get_new_position(row, col, action)

        # Check if landing on opponent's token
        if self.grid[new_row, new_col] == -self.current_player:
            # Capture opponent's token
            self.grid[row, col] = 0
            self.grid[new_row, new_col] = self.current_player
            self.player_positions[self.current_player] = (new_row, new_col)
            self.player_positions[-self.current_player] = (
                None  # Opponent token captured
            )
            self.done = True
            return self.grid.copy(), 1, True, False, {}

        # Move to new position
        self.grid[row, col] = 0
        self.grid[new_row, new_col] = self.current_player
        self.player_positions[self.current_player] = (new_row, new_col)

        # Check for victory by reaching opponent's baseline
        if (self.current_player == 1 and new_row == 4) or (
            self.current_player == -1 and new_row == 0
        ):
            self.done = True
            return self.grid.copy(), 1, True, False, {}

        # Switch player
        self.current_player *= -1
        return self.grid.copy(), 0, False, False, {}

    def render(self):
        grid_str = ""
        for r in range(5):
            row_str = ""
            for c in range(5):
                if self.grid[r, c] == 1:
                    row_str += " P1 "
                elif self.grid[r, c] == -1:
                    row_str += " P2 "
                else:
                    row_str += "  . "
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        moves = []
        if self.done:
            return moves  # No valid moves if game is over

        row, col = self.player_positions[self.current_player]

        # Define movement directions based on player
        if self.current_player == 1:
            movements = {
                0: (1, 0),  # Move Forward
                1: (0, -1),  # Move Left
                2: (0, 1),  # Move Right
                3: (1, -1),  # Diagonal Forward Left
                4: (1, 1),  # Diagonal Forward Right
            }
        else:
            movements = {
                0: (-1, 0),  # Move Forward
                1: (0, -1),  # Move Left
                2: (0, 1),  # Move Right
                3: (-1, -1),  # Diagonal Forward Left
                4: (-1, 1),  # Diagonal Forward Right
            }

        for action, (dr, dc) in movements.items():
            new_row, new_col = row + dr, col + dc

            # Stay within grid bounds
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                # No backward moves
                if self.current_player == 1 and dr < 0:
                    continue
                if self.current_player == -1 and dr > 0:
                    continue
                moves.append(action)

        return moves

    def get_new_position(self, row, col, action):
        if self.current_player == 1:
            movements = {
                0: (1, 0),  # Move Forward
                1: (0, -1),  # Move Left
                2: (0, 1),  # Move Right
                3: (1, -1),  # Diagonal Forward Left
                4: (1, 1),  # Diagonal Forward Right
            }
        else:
            movements = {
                0: (-1, 0),  # Move Forward
                1: (0, -1),  # Move Left
                2: (0, 1),  # Move Right
                3: (-1, -1),  # Diagonal Forward Left
                4: (-1, 1),  # Diagonal Forward Right
            }

        dr, dc = movements[action]
        new_row, new_col = row + dr, col + dc
        return new_row, new_col
