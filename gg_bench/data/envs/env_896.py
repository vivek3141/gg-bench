import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Action space:
        # Actions 0-3: Movement actions
        # 0: Move Up
        # 1: Move Down
        # 2: Move Left
        # 3: Move Right
        # Actions 4-28: Place obstacle at cell index (action - 4)

        self.action_space = spaces.Discrete(29)

        # Observation space: 5x5 grid
        # 0: empty
        # 1: current player's token
        # -1: opponent's token
        # 2: obstacle

        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid
        self.grid = np.zeros((5, 5), dtype=np.int8)

        # Place the players
        self.grid[2, 0] = 1  # P1 starts at A3 (row 2, col 0)
        self.grid[2, 4] = -1  # P2 starts at E3 (row 2, col 4)

        # Store player positions
        self.p1_pos = (2, 0)
        self.p2_pos = (2, 4)

        # Set current player: 1 or -1
        self.current_player = 1  # P1 starts first

        # Game done flag
        self.done = False

        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, return
            return self.grid.copy(), 0, True, False, {}

        if not self.action_space.contains(action):
            # Invalid action
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        reward = 0

        if action >= 0 and action <= 3:
            # Movement action
            # Get current player's position
            if self.current_player == 1:
                current_pos = self.p1_pos
            else:
                current_pos = self.p2_pos

            # Compute new position
            row, col = current_pos
            if action == 0:  # Move Up
                new_row, new_col = row - 1, col
            elif action == 1:  # Move Down
                new_row, new_col = row + 1, col
            elif action == 2:  # Move Left
                new_row, new_col = row, col - 1
            elif action == 3:  # Move Right
                new_row, new_col = row, col + 1

            # Check if new position is within bounds
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                # Check if new position is empty
                cell_value = self.grid[new_row, new_col]

                if cell_value == 0:
                    # Move is valid
                    # Update grid
                    self.grid[row, col] = 0  # Clear old position
                    self.grid[new_row, new_col] = self.current_player

                    # Update player's position
                    if self.current_player == 1:
                        self.p1_pos = (new_row, new_col)
                    else:
                        self.p2_pos = (new_row, new_col)

                    # Check for win condition
                    if self.current_player == 1 and new_col == 4:
                        # P1 reaches opponent's edge
                        reward = 1
                        self.done = True
                        return self.grid.copy(), reward, True, False, {}
                    elif self.current_player == -1 and new_col == 0:
                        # P2 reaches opponent's edge
                        reward = 1
                        self.done = True
                        return self.grid.copy(), reward, True, False, {}

                else:
                    # Invalid move (cell occupied)
                    reward = -10
                    self.done = True
                    return self.grid.copy(), reward, True, False, {}
            else:
                # Invalid move (out of bounds)
                reward = -10
                self.done = True
                return self.grid.copy(), reward, True, False, {}
        elif action >= 4 and action <= 28:
            # Obstacle placement action
            # Compute cell index
            cell_index = action - 4
            row = cell_index // 5
            col = cell_index % 5

            # Check if cell is empty
            if self.grid[row, col] == 0:
                # Check if cell is adjacent to opponent's position
                if self.current_player == 1:
                    opponent_pos = self.p2_pos
                else:
                    opponent_pos = self.p1_pos

                opp_row, opp_col = opponent_pos

                adjacent = False
                if abs(row - opp_row) + abs(col - opp_col) == 1:
                    adjacent = True

                if not adjacent:
                    # Valid obstacle placement
                    self.grid[row, col] = 2  # Obstacle

                else:
                    # Invalid placement (adjacent to opponent)
                    reward = -10
                    self.done = True
                    return self.grid.copy(), reward, True, False, {}
            else:
                # Invalid placement (cell occupied)
                reward = -10
                self.done = True
                return self.grid.copy(), reward, True, False, {}
        else:
            # Invalid action
            reward = -10
            self.done = True
            return self.grid.copy(), reward, True, False, {}

        # Switch current player
        self.current_player *= -1

        # Return observation, reward, done, truncated, info
        return self.grid.copy(), reward, self.done, False, {}

    def render(self):
        grid_repr = "    A   B   C   D   E\n"
        grid_repr += "  +---+---+---+---+---+\n"
        for i in range(5):
            grid_repr += f"{i+1} |"
            for j in range(5):
                cell = self.grid[i, j]
                if cell == 0:
                    grid_repr += "   |"
                elif cell == 1:
                    grid_repr += "P1 |"
                elif cell == -1:
                    grid_repr += "P2 |"
                elif cell == 2:
                    grid_repr += " X |"
            grid_repr += "\n  +---+---+---+---+---+\n"
        return grid_repr

    def valid_moves(self):
        valid_actions = []

        # Movement actions (0-3)
        if self.current_player == 1:
            row, col = self.p1_pos
        else:
            row, col = self.p2_pos

        # For each possible movement, check if it's valid
        for action in range(4):
            if action == 0:  # Move Up
                new_row, new_col = row - 1, col
            elif action == 1:  # Move Down
                new_row, new_col = row + 1, col
            elif action == 2:  # Move Left
                new_row, new_col = row, col - 1
            elif action == 3:  # Move Right
                new_row, new_col = row, col + 1

            if 0 <= new_row < 5 and 0 <= new_col < 5:
                cell_value = self.grid[new_row, new_col]
                if cell_value == 0:
                    valid_actions.append(action)

        # Obstacle placement actions (4-28)
        if self.current_player == 1:
            opp_row, opp_col = self.p2_pos
        else:
            opp_row, opp_col = self.p1_pos

        for action in range(4, 29):
            cell_index = action - 4
            row = cell_index // 5
            col = cell_index % 5

            if self.grid[row, col] == 0:
                # Check if cell is adjacent to opponent's position
                adjacent = False
                if abs(row - opp_row) + abs(col - opp_col) == 1:
                    adjacent = True
                if not adjacent:
                    valid_actions.append(action)

        return valid_actions
