import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.grid_size = 5
        self.action_space = spaces.Discrete(4)

        # The grid is represented by a 5x5 matrix with values:
        # 0 - empty cell
        # 1 - Player 1's token
        # 2 - Player 2's token
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.grid_size, self.grid_size), dtype=np.int8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        # Initial positions
        self.p1_pos = [0, 0]  # Player 1 starts at (0,0)
        self.p2_pos = [4, 4]  # Player 2 starts at (4,4)

        # Place the players on the grid
        self.grid[self.p1_pos[0], self.p1_pos[1]] = 1
        self.grid[self.p2_pos[0], self.p2_pos[1]] = 2

        self.current_player = 1  # Player 1 starts
        self.done = False
        info = {}
        return self.grid.copy(), info

    def step(self, action):
        if self.done:
            # Invalid move: game is already over
            return self.grid.copy(), -10, True, False, {}

        # Map action to movement
        action_deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        if action not in action_deltas:
            # Invalid action index
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        delta_row, delta_col = action_deltas[action]

        # Get current player's position and goal
        if self.current_player == 1:
            curr_row, curr_col = self.p1_pos
            opponent_row, opponent_col = self.p2_pos
            goal_row, goal_col = 4, 4  # Player 1's goal position
        else:
            curr_row, curr_col = self.p2_pos
            opponent_row, opponent_col = self.p1_pos
            goal_row, goal_col = 0, 0  # Player 2's goal position

        new_row = curr_row + delta_row
        new_col = curr_col + delta_col

        # Check if move is within grid bounds
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            # Invalid move: off the grid
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Check if new position is occupied by opponent
        if new_row == opponent_row and new_col == opponent_col:
            # Invalid move: square occupied by opponent
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Move is valid
        # Update the grid
        self.grid[curr_row, curr_col] = 0  # Remove token from old position
        self.grid[new_row, new_col] = self.current_player  # Place token in new position

        # Update player's position
        if self.current_player == 1:
            self.p1_pos = [new_row, new_col]
        else:
            self.p2_pos = [new_row, new_col]

        # Check for victory
        if new_row == goal_row and new_col == goal_col:
            # Current player wins
            self.done = True
            return self.grid.copy(), 1, True, False, {}

        # Switch to next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Continue the game
        return self.grid.copy(), 0, False, False, {}

    def render(self):
        grid_str = ""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = self.grid[row, col]
                if cell == 0:
                    grid_str += "- "
                elif cell == 1:
                    grid_str += "P1 "
                else:
                    grid_str += "P2 "
            grid_str += "\n"
        return grid_str

    def valid_moves(self):
        valid_actions = []
        action_deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        if self.done:
            return []

        # Get current player's position and opponent's position
        if self.current_player == 1:
            curr_row, curr_col = self.p1_pos
            opponent_pos = self.p2_pos
        else:
            curr_row, curr_col = self.p2_pos
            opponent_pos = self.p1_pos

        for action, (delta_row, delta_col) in action_deltas.items():
            new_row = curr_row + delta_row
            new_col = curr_col + delta_col
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                if [new_row, new_col] != opponent_pos:
                    valid_actions.append(action)

        return valid_actions
