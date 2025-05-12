import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 4 possible moves (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)

        # Define observation space: 7x7 grid with integer values
        # Values can be -2 (Player 2's trail), -1 (Player 2), 0 (empty), 1 (Player 1), 2 (Player 1's trail)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(7, 7), dtype=np.int8)

        # Initialize the game state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid
        self.grid = np.zeros((7, 7), dtype=np.int8)

        # Set the starting positions
        self.p1_pos = (0, 0)  # Player 1 starts at top-left corner
        self.p2_pos = (6, 6)  # Player 2 starts at bottom-right corner

        # Update the grid with the starting positions
        self.grid[self.p1_pos] = 1  # Player 1
        self.grid[self.p2_pos] = -1  # Player 2

        # Set the current player
        self.current_player = 1  # Player 1 starts

        self.done = False  # Game is not over

        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        # Mapping of actions to movements
        move_mapping = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        if action not in move_mapping:
            # Invalid action
            return self.grid.copy(), -10, True, False, {}

        dx, dy = move_mapping[action]

        # Get current player's position
        if self.current_player == 1:
            x, y = self.p1_pos
            opponent_pos = self.p2_pos
            opponent_start_pos = (6, 6)
            trail_value = 2  # Player 1's trail
            opponent = -1
            opponent_trail = -2
        else:  # self.current_player == -1
            x, y = self.p2_pos
            opponent_pos = self.p1_pos
            opponent_start_pos = (0, 0)
            trail_value = -2  # Player 2's trail
            opponent = 1
            opponent_trail = 2

        new_x = x + dx
        new_y = y + dy

        # Check if new position is within bounds
        if not (0 <= new_x < 7 and 0 <= new_y < 7):
            # Move is off the grid
            return self.grid.copy(), -10, True, False, {}

        # Check if new position is empty
        cell_value = self.grid[new_x, new_y]
        if cell_value != 0:
            # Cell is occupied
            return self.grid.copy(), -10, True, False, {}

        # Move is valid; update the grid
        # Mark the previous position as trail
        self.grid[x, y] = trail_value

        # Update player's position
        self.grid[new_x, new_y] = self.current_player
        if self.current_player == 1:
            self.p1_pos = (new_x, new_y)
        else:
            self.p2_pos = (new_x, new_y)

        # Check for victory conditions
        reward = 0
        self.done = False
        if (new_x, new_y) == opponent_start_pos:
            # Player has reached opponent's starting position
            reward = 1
            self.done = True
        else:
            # Check if opponent has any valid moves
            opponent_moves = self.get_valid_moves(opponent)
            if not opponent_moves:
                # Opponent has no valid moves
                reward = 1
                self.done = True

        # Switch current player
        if not self.done:
            # Only switch players if game is not over
            self.current_player *= -1

        return self.grid.copy(), reward, self.done, False, {}

    def get_valid_moves(self, player):
        # Get the player's current position
        if player == 1:
            x, y = self.p1_pos
        else:
            x, y = self.p2_pos

        # Possible moves: Up, Down, Left, Right
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for idx, (dx, dy) in enumerate(directions):
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < 7 and 0 <= new_y < 7:
                cell_value = self.grid[new_x, new_y]
                if cell_value == 0:
                    moves.append(idx)
        return moves

    def valid_moves(self):
        # Returns list of valid actions for current player
        return self.get_valid_moves(self.current_player)

    def render(self):
        # Build the string representation of the grid
        grid_repr = ""
        symbol_mapping = {0: ".", 1: "P1", 2: "1", -1: "P2", -2: "2"}
        for i in range(7):
            row = ""
            for j in range(7):
                cell_value = self.grid[i, j]
                symbol = symbol_mapping.get(cell_value, "?")
                row += f"{symbol} "
            grid_repr += row.strip() + "\n"
        return grid_repr.strip()
