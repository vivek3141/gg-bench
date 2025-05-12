import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: 32 possible actions (4 move directions x 8 block placements)
        self.action_space = spaces.Discrete(32)

        # Observation space: 5x5 grid with values:
        # 0: Empty, 1: Player 1 marker, 2: Player 2 marker, 3: Block
        self.observation_space = spaces.Box(low=0, high=3, shape=(5, 5), dtype=np.int8)

        # Movement and block placement mappings
        self.move_mapping = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }
        self.block_mapping = {
            0: (-1, 0),  # N
            1: (-1, 1),  # NE
            2: (0, 1),  # E
            3: (1, 1),  # SE
            4: (1, 0),  # S
            5: (1, -1),  # SW
            6: (0, -1),  # W
            7: (-1, -1),  # NW
        }

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid
        # 0: Empty
        # 1: Player 1 marker
        # 2: Player 2 marker
        # 3: Block
        self.grid = np.zeros((5, 5), dtype=np.int8)
        # Place player markers
        self.grid[0, 0] = 1  # Player 1 at (0, 0)
        self.grid[4, 4] = 2  # Player 2 at (4, 4)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.grid, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid, 0, True, False, {}

        move_direction = action // 8
        block_direction = action % 8

        # Get current position of the current player
        pos = np.argwhere(self.grid == self.current_player)
        if len(pos) == 0:
            return self.grid, -10, True, False, {}
        row, col = pos[0]

        # Calculate new position after move
        delta_row, delta_col = self.move_mapping[move_direction]
        new_row, new_col = row + delta_row, col + delta_col

        # Check move validity
        if not (0 <= new_row < 5 and 0 <= new_col < 5):
            self.done = True
            reward = -10
            return self.grid, reward, True, False, {}
        if self.grid[new_row, new_col] != 0:
            self.done = True
            reward = -10
            return self.grid, reward, True, False, {}

        # Move the marker
        self.grid[row, col] = 0  # Empty the previous cell
        self.grid[new_row, new_col] = self.current_player

        # Place block
        delta_row_b, delta_col_b = self.block_mapping[block_direction]
        block_row, block_col = new_row + delta_row_b, new_col + delta_col_b

        # Check block placement validity
        if not (0 <= block_row < 5 and 0 <= block_col < 5):
            self.done = True
            reward = -10
            return self.grid, reward, True, False, {}
        if self.grid[block_row, block_col] != 0:
            self.done = True
            reward = -10
            return self.grid, reward, True, False, {}

        # Place the block
        self.grid[block_row, block_col] = 3  # 3 represents block

        # Check if opponent can move
        opponent = 2 if self.current_player == 1 else 1
        opponent_pos = np.argwhere(self.grid == opponent)
        if len(opponent_pos) == 0:
            self.done = True
            reward = 1  # Current player wins
            return self.grid, reward, True, False, {}
        opp_row, opp_col = opponent_pos[0]

        # Check if opponent has any valid moves
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        opponent_can_move = False
        for d_row, d_col in directions:
            o_new_row, o_new_col = opp_row + d_row, opp_col + d_col
            if 0 <= o_new_row < 5 and 0 <= o_new_col < 5:
                if self.grid[o_new_row, o_new_col] == 0:
                    opponent_can_move = True
                    break
        if not opponent_can_move:
            self.done = True
            reward = 1
            return self.grid, reward, True, False, {}

        # Switch to the next player
        self.current_player = opponent

        return (
            self.grid,
            0,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        grid_str = "   0  1  2  3  4\n"
        symbols = {0: ".", 1: "P1", 2: "P2", 3: " X"}
        for i in range(5):
            grid_str += f"{i} ["
            for j in range(5):
                val = self.grid[i, j]
                grid_str += f"{symbols[val]:>3}"
            grid_str += " ]\n"
        return grid_str

    def valid_moves(self):
        if self.done:
            return []

        valid_actions = []

        # Get current position of the current player
        pos = np.argwhere(self.grid == self.current_player)
        if len(pos) == 0:
            return []

        row, col = pos[0]

        # For each possible move direction
        for move_dir in range(4):
            delta_row, delta_col = self.move_mapping[move_dir]
            new_row, new_col = row + delta_row, col + delta_col

            if not (0 <= new_row < 5 and 0 <= new_col < 5):
                continue
            if self.grid[new_row, new_col] != 0:
                continue  # Can't move here

            # From the new position, list possible block placements
            for block_dir in range(8):
                delta_row_b, delta_col_b = self.block_mapping[block_dir]
                block_row, block_col = new_row + delta_row_b, new_col + delta_col_b

                if not (0 <= block_row < 5 and 0 <= block_col < 5):
                    continue
                if self.grid[block_row, block_col] != 0:
                    continue  # Can't place block here

                # Action is valid
                action = move_dir * 8 + block_dir
                valid_actions.append(action)

        return valid_actions
