import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Up, 1 - Down, 2 - Left, 3 - Right
        self.action_space = spaces.Discrete(4)

        # Define observation space: 5x5 grid with values 0 (unvisited), 1 (visited), 2 (token position)
        self.observation_space = spaces.Box(low=0, high=2, shape=(5, 5), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int32)
        self.token_position = [2, 2]  # Starting at cell (3,3) in 1-based indexing
        self.grid[self.token_position[0], self.token_position[1]] = (
            2  # Mark the token position
        )
        self.current_player = 1
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}  # Game is over

        # Map actions to movements
        move_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        if action not in move_map:
            return self.grid.copy(), -10, True, False, {"invalid_move": True}

        move = move_map[action]
        new_row = self.token_position[0] + move[0]
        new_col = self.token_position[1] + move[1]

        # Check if new position is within bounds
        if not (0 <= new_row < 5 and 0 <= new_col < 5):
            return self.grid.copy(), -10, True, False, {"invalid_move": True}

        # Check if new position is an unvisited cell
        if self.grid[new_row, new_col] == 1:
            return self.grid.copy(), -10, True, False, {"invalid_move": True}

        # Valid move, proceed with updating the grid
        # Mark the previous cell as visited
        self.grid[self.token_position[0], self.token_position[1]] = 1  # Mark as visited

        # Move the token to the new position
        self.token_position = [new_row, new_col]
        self.grid[new_row, new_col] = 2  # Set token in new position

        # Check if the token is on an edge cell
        if new_row == 0 or new_row == 4 or new_col == 0 or new_col == 4:
            self.done = True
            reward = 1  # Current player wins
            return self.grid.copy(), reward, True, False, {}

        # Switch current player
        self.current_player *= -1

        # Check if the next player has valid moves
        if not self.has_valid_moves():
            self.done = True
            reward = 1  # Current player wins because opponent cannot move
            return self.grid.copy(), reward, True, False, {}

        # Valid move, game continues
        reward = -10  # Penalty for making a valid move (to encourage quick wins)
        return (
            self.grid.copy(),
            reward,
            False,
            False,
            {},
        )  # Observation, reward, done, truncated, info

    def render(self):
        # Generate a string representation of the grid
        grid_str = "    1   2   3   4   5\n"  # Column headers
        grid_str += "  +" + "---+" * 5 + "\n"  # Top border
        for i in range(5):
            grid_str += f"{i+1} |"
            for j in range(5):
                cell = self.grid[i, j]
                if cell == 0:
                    grid_str += "   |"
                elif cell == 1:
                    grid_str += " X |"  # Visited cell
                elif cell == 2:
                    grid_str += " T |"  # Token position
            grid_str += "\n  +" + "---+" * 5 + "\n"
        return grid_str

    def valid_moves(self):
        # Return a list of valid moves from the current token position
        moves = []
        move_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }
        for action in move_map:
            move = move_map[action]
            new_row = self.token_position[0] + move[0]
            new_col = self.token_position[1] + move[1]
            # Check if new position is within bounds
            if not (0 <= new_row < 5 and 0 <= new_col < 5):
                continue
            # Check if new position is an unvisited cell
            if self.grid[new_row, new_col] == 1:
                continue
            moves.append(action)
        return moves

    def has_valid_moves(self):
        # Check if there are any valid moves available
        return len(self.valid_moves()) > 0
