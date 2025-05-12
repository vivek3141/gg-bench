import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 50 actions: 25 move actions and 25 block actions
        self.action_space = spaces.Discrete(50)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=int)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=int)
        # Place P1 at (0,2) and P2 at (4,2)
        self.grid[0, 2] = 1  # P1
        self.grid[4, 2] = 2  # P2
        self.current_player = 1  # 1 for P1, 2 for P2
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), -10, True, False, {}
        reward = 0

        # Determine action type and target cell
        if action < 25:
            # Move action
            action_type = "move"
            target_index = action
        else:
            # Block action
            action_type = "block"
            target_index = action - 25

        target_row = target_index // 5
        target_col = target_index % 5

        if action_type == "move":
            valid = self.valid_move_action(target_row, target_col)
            if not valid:
                self.done = True
                return self.grid.copy(), -10, True, False, {}
            else:
                # Move the player's piece
                self.execute_move(target_row, target_col)
        elif action_type == "block":
            valid = self.valid_block_action(target_row, target_col)
            if not valid:
                self.done = True
                return self.grid.copy(), -10, True, False, {}
            else:
                # Place the block
                self.grid[target_row, target_col] = -1

        # Check for win conditions
        if self.check_win():
            reward = 1
            self.done = True
            return self.grid.copy(), reward, True, False, {}
        else:
            # Check if the opponent has any valid moves
            self.current_player = 3 - self.current_player  # Switch player
            if not self.has_valid_moves():
                reward = 1
                self.done = True
                return self.grid.copy(), reward, True, False, {}
            else:
                return self.grid.copy(), reward, False, False, {}

    def render(self):
        """Return a string representation of the current grid."""
        symbols = {0: ".", 1: "P1", 2: "P2", -1: "X"}
        grid_str = ""
        for r in range(5):
            row_str = ""
            for c in range(5):
                cell = self.grid[r, c]
                row_str += f"{symbols[cell]:^3} "
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        """Return a list of valid action indices."""
        valid_actions = []
        for action in range(50):
            if action < 25:
                # Move actions
                target_row = action // 5
                target_col = action % 5
                if self.valid_move_action(target_row, target_col):
                    valid_actions.append(action)
            else:
                # Block actions
                target_row = (action - 25) // 5
                target_col = (action - 25) % 5
                if self.valid_block_action(target_row, target_col):
                    valid_actions.append(action)
        return valid_actions

    def valid_move_action(self, target_row, target_col):
        if not (0 <= target_row < 5 and 0 <= target_col < 5):
            return False
        if self.grid[target_row, target_col] != 0:
            return False
        current_row, current_col = self.get_current_player_position()
        if abs(target_row - current_row) <= 1 and abs(target_col - current_col) <= 1:
            return True
        else:
            return False

    def valid_block_action(self, target_row, target_col):
        if not (0 <= target_row < 5 and 0 <= target_col < 5):
            return False
        if self.grid[target_row, target_col] != 0:
            return False
        p1_row, p1_col = self.get_player_position(1)
        p2_row, p2_col = self.get_player_position(2)
        if abs(target_row - p1_row) <= 1 and abs(target_col - p1_col) <= 1:
            return False
        if abs(target_row - p2_row) <= 1 and abs(target_col - p2_col) <= 1:
            return False
        return True

    def get_current_player_position(self):
        return self.get_player_position(self.current_player)

    def get_player_position(self, player):
        positions = np.argwhere(self.grid == player)
        if len(positions) > 0:
            return positions[0][0], positions[0][1]
        else:
            return None, None

    def execute_move(self, target_row, target_col):
        current_row, current_col = self.get_current_player_position()
        self.grid[current_row, current_col] = 0
        self.grid[target_row, target_col] = self.current_player

    def check_win(self):
        target_row = 0 if self.current_player == 2 else 4
        player_row, _ = self.get_current_player_position()
        if player_row == target_row:
            return True
        return False

    def has_valid_moves(self):
        # Check if the opponent has any valid moves
        opponent = self.current_player
        moves_available = False
        for action in range(50):
            if action < 25:
                # Move actions
                target_row = action // 5
                target_col = action % 5
                if self.current_player == opponent and self.valid_move_action(
                    target_row, target_col
                ):
                    moves_available = True
                    break
            else:
                # Block actions
                target_row = (action - 25) // 5
                target_col = (action - 25) % 5
                if self.current_player == opponent and self.valid_block_action(
                    target_row, target_col
                ):
                    moves_available = True
                    break
        self.current_player = opponent  # Switch back to original player
        return moves_available
