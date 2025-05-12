import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(25,), dtype=np.int32)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int32)
        self.token_pos = (2, 2)  # Token starts at the center
        self.grid[self.token_pos] = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        # Track the last positions for each player to prevent immediate backtracking
        self.last_positions = {1: None, -1: None}
        return self.grid.flatten(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.flatten(), 0, True, False, {}  # Game is over

        # Map action to direction
        direction = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        move = direction[action]

        new_row = self.token_pos[0] + move[0]
        new_col = self.token_pos[1] + move[1]
        new_pos = (new_row, new_col)

        # Check for invalid moves (outside the grid boundaries)
        if not (0 <= new_row <= 4 and 0 <= new_col <= 4):
            self.done = True
            return self.grid.flatten(), -10, True, False, {}

        # Check for prohibited move (immediate backtracking)
        opponent = -self.current_player
        if self.last_positions[opponent] == new_pos:
            self.done = True
            return self.grid.flatten(), -10, True, False, {}

        # Move the token
        self.grid[self.token_pos] = 0
        self.token_pos = new_pos
        self.grid[self.token_pos] = 1

        # Update last position for the current player
        self.last_positions[self.current_player] = self.token_pos

        # Check for win condition
        if self.current_player == 1 and self.token_pos == (4, 4):
            self.done = True
            return self.grid.flatten(), 1, True, False, {}
        if self.current_player == -1 and self.token_pos == (0, 0):
            self.done = True
            return self.grid.flatten(), 1, True, False, {}

        # Switch current player
        self.current_player *= -1

        # Continue the game
        return self.grid.flatten(), 0, False, False, {}

    def render(self):
        grid_str = ""
        for i in range(5):
            for j in range(5):
                if (i, j) == self.token_pos:
                    grid_str += "[T] "
                elif (i, j) == (0, 0):
                    grid_str += "[P1] "
                elif (i, j) == (4, 4):
                    grid_str += "[P2] "
                else:
                    grid_str += ".   "
            grid_str += "\n"
        return grid_str

    def valid_moves(self):
        valid_actions = []
        direction = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        opponent = -self.current_player
        for action in range(4):
            move = direction[action]
            new_row = self.token_pos[0] + move[0]
            new_col = self.token_pos[1] + move[1]
            new_pos = (new_row, new_col)
            # Check grid boundaries
            if not (0 <= new_row <= 4 and 0 <= new_col <= 4):
                continue
            # Check for prohibited immediate backtracking
            if self.last_positions[opponent] == new_pos:
                continue
            valid_actions.append(action)
        return valid_actions
