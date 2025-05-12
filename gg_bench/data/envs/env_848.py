import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation spaces
        self.action_space = spaces.Discrete(545)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)
        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # 1 for Player 1 (X), -1 for Player 2 (O)
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}  # No reward, game is over

        # Initialize reward
        reward = 0

        # Map action_index to placement_cell_index and shift_option_index
        if action < 525:
            # Normal action: placement + optional shift
            placement_cell_index = action // 21
            shift_option_index = action % 21
            # placement_cell_index ranges from 0 to 24
            # shift_option_index ranges from 0 to 20
        elif action < 545:
            # Shift-only action when grid is full
            placement_cell_index = -1  # Indicates no placement
            shift_option_index = action - 524  # shift_option_index ranges from 1 to 20
        else:
            # Invalid action index
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Check if grid is full
        grid_full = not (self.grid == 0).any()

        if grid_full:
            if placement_cell_index != -1:
                # Invalid action: cannot place a marker when grid is full
                self.done = True
                return self.grid.copy(), -10, True, False, {}
        else:
            if placement_cell_index == -1:
                # Invalid action: must place a marker when grid is not full
                self.done = True
                return self.grid.copy(), -10, True, False, {}
            # Map placement_cell_index to (row, col)
            row = placement_cell_index // 5
            col = placement_cell_index % 5
            # Check if cell is empty
            if self.grid[row, col] != 0:
                # Invalid move: cell is already occupied
                self.done = True
                return self.grid.copy(), -10, True, False, {}
            # Place marker
            self.grid[row, col] = self.current_player

        # Perform the shift operation, if any
        if shift_option_index != 0:
            if 1 <= shift_option_index <= 10:
                # Row shifts
                shift_index = shift_option_index - 1
                row_num = shift_index // 2  # row_num from 0 to 4
                direction = "left" if shift_index % 2 == 0 else "right"
                # Perform the shift
                if direction == "left":
                    self.grid[row_num] = np.roll(self.grid[row_num], -1)
                else:  # direction == 'right'
                    self.grid[row_num] = np.roll(self.grid[row_num], 1)
            elif 11 <= shift_option_index <= 20:
                # Column shifts
                shift_index = shift_option_index - 11
                col_num = shift_index // 2  # col_num from 0 to 4
                direction = "up" if shift_index % 2 == 0 else "down"
                # Perform the shift
                if direction == "up":
                    self.grid[:, col_num] = np.roll(self.grid[:, col_num], -1)
                else:  # direction == 'down'
                    self.grid[:, col_num] = np.roll(self.grid[:, col_num], 1)
            else:
                # Invalid shift_option_index
                self.done = True
                return self.grid.copy(), -10, True, False, {}
        # else shift_option_index == 0: No shift

        # Check if current player has won
        if self.check_win(self.current_player):
            reward = 1
            self.done = True
        else:
            # Game continues
            reward = 0
            # Switch to the next player
            self.current_player *= -1  # Switch between 1 and -1

        return (
            self.grid.copy(),
            reward,
            self.done,
            False,
            {},
        )  # observation, reward, terminated, truncated, info

    def check_win(self, player):
        # Check rows and columns for a line of three
        for i in range(5):
            # Check rows
            for j in range(3):
                if np.all(self.grid[i, j : j + 3] == player):
                    return True
            # Check columns
            for j in range(3):
                if np.all(self.grid[j : j + 3, i] == player):
                    return True
        # Check diagonals (left-to-right)
        for i in range(3):
            for j in range(3):
                if (
                    self.grid[i, j] == player
                    and self.grid[i + 1, j + 1] == player
                    and self.grid[i + 2, j + 2] == player
                ):
                    return True
        # Check diagonals (right-to-left)
        for i in range(3):
            for j in range(2, 5):
                if (
                    self.grid[i, j] == player
                    and self.grid[i + 1, j - 1] == player
                    and self.grid[i + 2, j - 2] == player
                ):
                    return True
        return False

    def render(self):
        board_str = "    A   B   C   D   E\n"
        board_str += "  +---+---+---+---+---+\n"
        for i in range(5):
            row_str = f"{i+1} |"
            for j in range(5):
                cell = self.grid[i, j]
                if cell == 1:
                    row_str += " X |"
                elif cell == -1:
                    row_str += " O |"
                else:
                    row_str += "   |"
            board_str += row_str + "\n"
            board_str += "  +---+---+---+---+---+\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        grid_full = not (self.grid == 0).any()
        if not grid_full:
            empty_cells = np.argwhere(self.grid == 0)
            for cell in empty_cells:
                cell_index = cell[0] * 5 + cell[1]  # Map (row, col) to cell_index
                for shift_option_index in range(
                    21
                ):  # shift_option_index ranges from 0 to 20
                    action_index = cell_index * 21 + shift_option_index
                    valid_actions.append(action_index)
        else:
            # Grid is full, only shift-only actions are valid
            for shift_option_index in range(
                1, 21
            ):  # shift_option_index ranges from 1 to 20
                action_index = 524 + shift_option_index  # action indices 525 to 544
                valid_actions.append(action_index)
        return valid_actions
