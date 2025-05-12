import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(49)
        self.observation_space = spaces.Box(low=0, high=6, shape=(5, 5), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int32)

        # Set the bases
        self.grid[2, 0] = 1  # Player 1's base at (1,3)
        self.grid[2, 4] = 2  # Player 2's base at (5,3)

        self.current_player = 1  # 1 for Player 1, 2 for Player 2
        self.done = False
        self.player_mirrors = {1: [], 2: []}  # Track mirrors for each player
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        # Actions 0-45: Place a mirror at a position with an orientation
        # Actions 46-48: Rotate one of your existing mirrors
        if action >= 0 and action <= 45:
            position_index = action // 2
            orientation = action % 2  # 0: '/', 1: '\'

            # Define the list of valid positions (excluding bases)
            position_list = [
                (0, 0),
                (1, 0),
                (3, 0),
                (4, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (3, 2),
                (4, 2),
                (0, 3),
                (1, 3),
                (2, 3),
                (3, 3),
                (4, 3),
                (0, 4),
                (1, 4),
                (3, 4),
                (4, 4),
            ]

            # Check for valid position index
            if position_index >= len(position_list):
                return self.grid.copy(), -10, True, False, {}

            row, col = position_list[position_index]

            # Check if the position is empty
            if self.grid[row, col] != 0:
                return self.grid.copy(), -10, True, False, {}

            # Check mirror limit
            if len(self.player_mirrors[self.current_player]) >= 3:
                return self.grid.copy(), -10, True, False, {}

            # Place the mirror
            if self.current_player == 1:
                cell_value = 3 if orientation == 0 else 4  # Player 1's mirror
            else:
                cell_value = 5 if orientation == 0 else 6  # Player 2's mirror

            self.grid[row, col] = cell_value
            self.player_mirrors[self.current_player].append((row, col))

        elif action >= 46 and action <= 48:
            mirror_index = action - 46

            # Check if the mirror exists
            if mirror_index >= len(self.player_mirrors[self.current_player]):
                return self.grid.copy(), -10, True, False, {}

            row, col = self.player_mirrors[self.current_player][mirror_index]
            cell_value = self.grid[row, col]

            # Rotate the mirror
            if cell_value in [3, 5]:  # '/' mirror
                new_cell_value = cell_value + 1  # Change to '\'
            else:  # '\' mirror
                new_cell_value = cell_value - 1  # Change to '/'

            self.grid[row, col] = new_cell_value

        else:
            return self.grid.copy(), -10, True, False, {}

        # Laser firing phase
        if self.current_player == 1:
            row, col, dir_r, dir_c = 2, 0, 0, 1  # Start from (1,3), moving right
            opponent_base_value = 2
        else:
            row, col, dir_r, dir_c = 2, 4, 0, -1  # Start from (5,3), moving left
            opponent_base_value = 1

        while True:
            row += dir_r
            col += dir_c

            # Check if laser exits the grid
            if row < 0 or row >= 5 or col < 0 or col >= 5:
                break

            cell = self.grid[row, col]

            if cell == opponent_base_value:
                # Laser hits the opponent's base: win
                self.done = True
                return self.grid.copy(), 1, True, False, {}

            elif cell == 0:
                # Empty square, continue laser path
                continue

            elif cell in [1, 2]:
                # Laser hits own base: ends
                break

            else:
                # Laser interacts with a mirror
                orientation = "/" if cell in [3, 5] else "\\"

                if orientation == "/":
                    if dir_r == 0 and dir_c == 1:  # Right
                        dir_r, dir_c = -1, 0  # Up
                    elif dir_r == 0 and dir_c == -1:  # Left
                        dir_r, dir_c = 1, 0  # Down
                    elif dir_r == 1 and dir_c == 0:  # Down
                        dir_r, dir_c = 0, -1  # Left
                    elif dir_r == -1 and dir_c == 0:  # Up
                        dir_r, dir_c = 0, 1  # Right
                    else:
                        break
                else:  # '\'
                    if dir_r == 0 and dir_c == 1:  # Right
                        dir_r, dir_c = 1, 0  # Down
                    elif dir_r == 0 and dir_c == -1:  # Left
                        dir_r, dir_c = -1, 0  # Up
                    elif dir_r == 1 and dir_c == 0:  # Down
                        dir_r, dir_c = 0, 1  # Right
                    elif dir_r == -1 and dir_c == 0:  # Up
                        dir_r, dir_c = 0, -1  # Left
                    else:
                        break

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1
        return self.grid.copy(), 0, False, False, {}

    def render(self):
        grid_str = ""
        for row in range(4, -1, -1):
            for col in range(5):
                cell = self.grid[row, col]
                if cell == 0:
                    grid_str += ". "
                elif cell == 1:
                    grid_str += "B1 "
                elif cell == 2:
                    grid_str += "B2 "
                elif cell == 3:
                    grid_str += "/1 "
                elif cell == 4:
                    grid_str += "\\1 "
                elif cell == 5:
                    grid_str += "/2 "
                elif cell == 6:
                    grid_str += "\\2 "
                else:
                    grid_str += "? "
            grid_str += "\n"
        return grid_str

    def valid_moves(self):
        moves = []
        if self.current_player == 1:
            mirror_values = [3, 4]
        else:
            mirror_values = [5, 6]

        # Possible mirror placements
        position_list = [
            (0, 0),
            (1, 0),
            (3, 0),
            (4, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
            (0, 4),
            (1, 4),
            (3, 4),
            (4, 4),
        ]

        if len(self.player_mirrors[self.current_player]) < 3:
            for idx, (row, col) in enumerate(position_list):
                if self.grid[row, col] == 0:
                    moves.append(idx * 2)
                    moves.append(idx * 2 + 1)

        # Possible rotations
        for mi in range(len(self.player_mirrors[self.current_player])):
            moves.append(46 + mi)

        return moves
