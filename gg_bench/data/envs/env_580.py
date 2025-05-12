import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            29
        )  # 4 movement actions + 25 block positions

        # Observation space: 5x5 grid
        # Values: -1 (blocked), 0 (empty), 1 (Player 1), 2 (Player 2)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid
        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.grid[0, 0] = 1  # Player 1 starting position
        self.grid[4, 4] = 2  # Player 2 starting position

        # Store player positions
        self.player_positions = {1: (0, 0), 2: (4, 4)}

        # Set the current player (Player 1 starts)
        self.current_player = 1
        self.opponent = 2

        self.terminated = False
        self.truncated = False

        return self.grid.copy(), {}  # observation, info

    def step(self, action):
        if self.terminated or self.truncated:
            return self.grid.copy(), -10, self.terminated, self.truncated, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            reward = -100
            self.terminated = True
            return self.grid.copy(), reward, self.terminated, self.truncated, {}

        # Default reward for a valid move
        reward = -10

        if action in [0, 1, 2, 3]:
            # Movement action
            dir_row, dir_col = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[action]
            row, col = self.player_positions[self.current_player]
            new_row, new_col = row + dir_row, col + dir_col

            # Move the player
            self.grid[row, col] = 0  # Remove player from old position
            self.player_positions[self.current_player] = (new_row, new_col)

            if (new_row, new_col) == ((0, 0) if self.opponent == 1 else (4, 4)):
                # Reached opponent's starting position
                self.grid[new_row, new_col] = self.current_player
                reward = 1
                self.terminated = True
                return self.grid.copy(), reward, self.terminated, self.truncated, {}

            else:
                # Move to the new position
                self.grid[new_row, new_col] = self.current_player

        else:
            # Block action
            block_idx = action - 4
            block_row, block_col = divmod(block_idx, 5)
            self.grid[block_row, block_col] = -1  # Place block

        # Switch turns
        self.current_player, self.opponent = self.opponent, self.current_player

        # Check if the next player can make any valid moves
        if not self.valid_moves():
            # Opponent cannot move; current player wins
            reward = 1
            self.terminated = True
            return self.grid.copy(), reward, self.terminated, self.truncated, {}

        return self.grid.copy(), reward, self.terminated, self.truncated, {}

    def render(self):
        # Create a string representation of the grid
        grid_str = ""
        for row in range(5):
            for col in range(5):
                cell = self.grid[row, col]
                if cell == -1:
                    grid_str += " X "
                elif cell == 0:
                    grid_str += " . "
                elif cell == 1:
                    grid_str += " P1"
                elif cell == 2:
                    grid_str += " P2"
            grid_str += "\n"
        return grid_str

    def valid_moves(self):
        valid_actions = []
        row, col = self.player_positions[self.current_player]

        # Movement actions
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        for action, (dr, dc) in moves.items():
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                if self.grid[new_row, new_col] == 0 or (new_row, new_col) == (
                    (0, 0) if self.opponent == 1 else (4, 4)
                ):
                    valid_actions.append(action)

        # Blocking actions
        for r in range(5):
            for c in range(5):
                if (
                    self.grid[r, c] == 0
                    and (r, c) != self.player_positions[1]
                    and (r, c) != self.player_positions[2]
                    and (r, c) != (0, 0)
                    and (r, c) != (4, 4)
                ):
                    action = 4 + r * 5 + c
                    valid_actions.append(action)

        return valid_actions
