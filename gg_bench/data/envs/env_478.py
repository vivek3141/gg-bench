import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define constants for cell contents
        self.EMPTY = 0
        self.BLOCKED = 1
        self.PLAYER_1_KNIGHT = 2
        self.PLAYER_2_KNIGHT = 3

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(25)  # 5x5 grid positions (0 to 24)
        self.observation_space = spaces.Box(low=0, high=3, shape=(5, 5), dtype=np.int8)

        # Initialize the environment state
        self.grid = None
        self.current_player = None  # 1 or 2
        self.player_positions = {}  # {1: (x, y), 2: (x, y)}
        self.terminated = False

        # Reset the environment to start
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid
        self.grid = np.full((5, 5), self.EMPTY, dtype=np.int8)

        # Set starting positions
        self.grid[0, 0] = self.PLAYER_1_KNIGHT
        self.grid[4, 4] = self.PLAYER_2_KNIGHT
        self.player_positions = {1: (0, 0), 2: (4, 4)}

        # Set the current player
        self.current_player = 1

        # Set the termination flag
        self.terminated = False

        return self.grid.copy(), {}

    def step(self, action):
        if self.terminated:
            return self.grid.copy(), 0, True, False, {}

        x_target = action % 5
        y_target = action // 5

        x_current, y_current = self.player_positions[self.current_player]

        # Calculate the move delta
        dx = x_target - x_current
        dy = y_target - y_current

        # Check if the move is a valid knight's move
        if (dx, dy) not in [
            (2, 1),
            (2, -1),
            (-2, 1),
            (-2, -1),
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
        ]:
            # Invalid move
            self.terminated = True
            return self.grid.copy(), -10, True, False, {}

        # Check if the target position is within bounds and empty
        if 0 <= x_target < 5 and 0 <= y_target < 5:
            if self.grid[y_target, x_target] == self.EMPTY:
                # Valid move
                # Block the previous position
                self.grid[y_current, x_current] = self.BLOCKED

                # Move the knight to the new position
                self.grid[y_target, x_target] = (
                    self.PLAYER_1_KNIGHT
                    if self.current_player == 1
                    else self.PLAYER_2_KNIGHT
                )

                # Update the player's knight position
                self.player_positions[self.current_player] = (x_target, y_target)

                # Switch to the opponent
                self.switch_player()

                # Check if the opponent has any valid moves
                opponent_moves = self.valid_moves()
                if not opponent_moves:
                    # Opponent cannot move, current player wins
                    # Switch back to the winner
                    self.switch_player()
                    self.terminated = True
                    return self.grid.copy(), 1, True, False, {}
                else:
                    # Game continues
                    return self.grid.copy(), 0, False, False, {}
            else:
                # Target cell is not empty
                self.terminated = True
                return self.grid.copy(), -10, True, False, {}
        else:
            # Target position is out of bounds
            self.terminated = True
            return self.grid.copy(), -10, True, False, {}

    def switch_player(self):
        self.current_player = 2 if self.current_player == 1 else 1

    def valid_moves(self):
        x, y = self.player_positions[self.current_player]
        possible_moves = []
        knight_moves = [
            (x + 2, y + 1),
            (x + 2, y - 1),
            (x - 2, y + 1),
            (x - 2, y - 1),
            (x + 1, y + 2),
            (x + 1, y - 2),
            (x - 1, y + 2),
            (x - 1, y - 2),
        ]

        for nx, ny in knight_moves:
            if 0 <= nx < 5 and 0 <= ny < 5:
                if self.grid[ny, nx] == self.EMPTY:
                    action = ny * 5 + nx
                    possible_moves.append(action)
        return possible_moves

    def render(self):
        # Map cell values to symbols
        symbol_map = {
            self.EMPTY: " . ",
            self.BLOCKED: " X ",
            self.PLAYER_1_KNIGHT: "K1 ",
            self.PLAYER_2_KNIGHT: "K2 ",
        }
        grid_str = ""
        for y in range(5):
            for x in range(5):
                cell_value = self.grid[y, x]
                grid_str += symbol_map[cell_value]
            grid_str += "\n"
        return grid_str
