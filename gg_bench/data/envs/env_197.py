import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Up, 1 - Down, 2 - Left, 3 - Right
        self.action_space = spaces.Discrete(4)

        # Observation space: 5x5 grid with values:
        # 0 - Empty, 1 - Blocked (X), 2 - Player 1 (A), 3 - Player 2 (B)
        self.observation_space = spaces.Box(low=0, high=3, shape=(5, 5), dtype=np.int8)

        # Grid parameters
        self.grid_size = 5

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Set starting positions
        self.player_positions = {
            1: (0, 0),  # Player 1 starts at top-left corner
            2: (4, 4),  # Player 2 starts at bottom-right corner
        }
        self.grid[0, 0] = 2  # Player 1 marker
        self.grid[4, 4] = 3  # Player 2 marker

        # Current player: 1 - Player 1, 2 - Player 2
        self.current_player = 1

        # Game over flag
        self.done = False

        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), -10, True, False, {}

        # Map action to movement
        action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Get current position
        row, col = self.player_positions[self.current_player]

        # Calculate new position
        dr, dc = action_map[action]
        new_row, new_col = row + dr, col + dc

        # Move the player
        self.grid[row, col] = 1  # Block the cell vacated
        self.player_positions[self.current_player] = (new_row, new_col)

        # Place the player's marker on the new cell
        if self.current_player == 1:
            self.grid[new_row, new_col] = 2  # Player 1 marker
        else:
            self.grid[new_row, new_col] = 3  # Player 2 marker

        reward = -10  # Default reward for a valid move

        # Check for victory conditions
        opponent_start = {1: (4, 4), 2: (0, 0)}
        opponent_player = 2 if self.current_player == 1 else 1

        # Victory Condition 1: Reaching opponent's starting position
        if (new_row, new_col) == opponent_start[self.current_player]:
            self.done = True
            reward = 1
            return self.grid.copy(), reward, True, False, {}

        # Victory Condition 2: Opponent has no legal moves
        self.current_player = opponent_player  # Switch to opponent for checking
        opponent_moves = self.valid_moves()
        self.current_player = opponent_player  # Keep the switch for the next turn

        if not opponent_moves:
            self.done = True
            reward = 1  # Current player wins
            return self.grid.copy(), reward, True, False, {}

        # Switch turn to the next player
        self.current_player = opponent_player

        return self.grid.copy(), reward, False, False, {}

    def render(self):
        # Create a visual representation of the grid
        symbols = {
            0: ".",
            1: "X",
            2: "A",
            3: "B",
        }
        grid_str = ""
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                cell = self.grid[i, j]
                row_str += f"[{symbols[cell]}] "
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        # Return a list of valid actions for the current player
        actions = []
        action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        # Get current position
        row, col = self.player_positions[self.current_player]

        for action, (dr, dc) in action_map.items():
            new_row, new_col = row + dr, col + dc

            # Check if the move is within bounds
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                # Check if the target cell is not blocked or occupied
                cell_value = self.grid[new_row, new_col]
                if cell_value == 0:
                    actions.append(action)
                elif cell_value == 1:
                    continue  # Blocked cell
                elif cell_value == 2 and self.current_player != 1:
                    continue  # Occupied by Player 1
                elif cell_value == 3 and self.current_player != 2:
                    continue  # Occupied by Player 2
        return actions
