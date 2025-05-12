import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 4 possible actions: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # Observation space is a flattened 5x5 grid
        self.observation_space = spaces.Box(low=0, high=1, shape=(25,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.token_position = (0, 0)  # Starting position at (1,1) which is index (0,0)
        self.grid[self.token_position] = 1
        self.last_move = None
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.grid.flatten(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.flatten(), -10, True, False, {}

        # Mapping from action index to direction
        action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        reverse_moves = {"up": "down", "down": "up", "left": "right", "right": "left"}

        direction = action_map.get(action)
        if direction is None:
            # Invalid action
            self.done = True
            return self.grid.flatten(), -10, True, False, {}

        # Check forbidden reversals
        if self.last_move is not None and direction == reverse_moves.get(
            self.last_move
        ):
            # Forbidden reversal
            self.done = True
            return self.grid.flatten(), -10, True, False, {}

        # Compute next position
        row, col = self.token_position
        if direction == "up":
            next_position = (row - 1, col)
        elif direction == "down":
            next_position = (row + 1, col)
        elif direction == "left":
            next_position = (row, col - 1)
        elif direction == "right":
            next_position = (row, col + 1)

        # Check grid boundaries
        if not (0 <= next_position[0] < 5 and 0 <= next_position[1] < 5):
            # Move would go off the grid
            self.done = True
            return self.grid.flatten(), -10, True, False, {}

        # Move the token
        self.grid[self.token_position] = 0
        self.token_position = next_position
        self.grid[self.token_position] = 1

        # Check for win condition
        if self.token_position == (2, 2):  # Center square at (3,3) which is index (2,2)
            self.done = True
            return self.grid.flatten(), 1, True, False, {}

        # Update last move
        self.last_move = direction

        # Switch player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        return self.grid.flatten(), 0, False, False, {}

    def render(self):
        grid_str = ""
        grid_str += "    1   2   3   4   5\n"
        grid_str += "  -------------------------\n"
        for i in range(5):
            grid_str += f"{i+1} |"
            for j in range(5):
                if self.grid[i][j] == 1:
                    grid_str += " * |"
                else:
                    grid_str += "   |"
            grid_str += "\n  -------------------------\n"
        return grid_str

    def valid_moves(self):
        # Return list of valid moves as action indices
        valid_actions = []
        row, col = self.token_position
        reverse_moves = {"up": "down", "down": "up", "left": "right", "right": "left"}
        directions = {"up": 0, "down": 1, "left": 2, "right": 3}

        for direction, action_idx in directions.items():
            # Check forbidden reversals
            if self.last_move is not None and direction == reverse_moves.get(
                self.last_move
            ):
                continue

            # Check grid boundaries
            if direction == "up" and row == 0:
                continue
            if direction == "down" and row == 4:
                continue
            if direction == "left" and col == 0:
                continue
            if direction == "right" and col == 4:
                continue

            valid_actions.append(action_idx)
        return valid_actions
