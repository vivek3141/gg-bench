import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is from 0 to 24 (25 positions on the grid)
        self.action_space = spaces.Discrete(25)
        # The observation will be the grid state, a 5x5 array
        # 0: empty cell, 1: Player 1's marker ('X'), -1: Player 2's marker ('O')
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        # Initialize the grid and game state
        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        # Convert action (0-24) to grid coordinates
        row = action // 5
        col = action % 5

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self.grid.copy(), reward, True, False, {}

        # Place the marker
        self.grid[row, col] = self.current_player

        # Check if opponent has any valid moves
        opponent_valid_actions = self._compute_valid_moves(-self.current_player)
        if not opponent_valid_actions:
            # Current player wins
            self.done = True
            reward = 1
            return self.grid.copy(), reward, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        reward = -10
        return self.grid.copy(), reward, False, False, {}

    def render(self):
        # Create a string representation of the grid
        symbols = {1: "X", -1: "O", 0: " "}
        rows = []
        for i in range(5):
            row = "|"
            for j in range(5):
                cell = symbols[self.grid[i, j]]
                row += f" {cell} |"
            rows.append(row)
        line = "+" + "----+" * 5
        board = "\n".join([line] + [f"{r}\n{line}" for r in rows])
        return board

    def valid_moves(self):
        return self._compute_valid_moves(self.current_player)

    def _compute_valid_moves(self, player):
        # Get opponent's markers
        opponent = -player
        opponent_positions = np.argwhere(self.grid == opponent)
        blocked_cells = set()

        # Compute blocked cells based on opponent's markers
        for op_row, op_col in opponent_positions:
            # Block the row
            for col in range(5):
                blocked_cells.add((op_row, col))
            # Block the column
            for row in range(5):
                blocked_cells.add((row, op_col))
            # Block the primary diagonal if applicable
            if op_row == op_col:
                for i in range(5):
                    blocked_cells.add((i, i))
            # Block the secondary diagonal if applicable
            if op_row + op_col == 4:
                for i in range(5):
                    blocked_cells.add((i, 4 - i))

        # Find all empty cells not blocked
        valid_actions = []
        for row in range(5):
            for col in range(5):
                if self.grid[row, col] == 0 and (row, col) not in blocked_cells:
                    action = row * 5 + col
                    valid_actions.append(action)
        return valid_actions
