import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Grid size
        self.grid_size = 5
        self.num_cells = self.grid_size * self.grid_size

        # Action space: 25 cells (0-24) + pass action (25)
        self.action_space = spaces.Discrete(
            self.num_cells + 1
        )  # Actions 0-24 for cells, 25 for pass

        # Observation space: 25 cells, values -1 (Player 2), 0 (empty), 1 (Player 1)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.num_cells,), dtype=np.int8
        )

        # Initialize the board and other variables
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(self.num_cells, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.consecutive_passes = 0  # Number of consecutive passes
        self.done = False
        self.info = {}
        return self.board.copy(), self.info  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, self.info

        if action < 0 or action > self.num_cells:
            # Invalid action index
            return self.board.copy(), -10, True, False, self.info

        if action == self.num_cells:
            # Pass action
            valid_moves = self.valid_moves()
            if len(valid_moves) > 0:
                # Cannot pass if there are valid moves
                return self.board.copy(), -10, True, False, self.info
            else:
                self.consecutive_passes += 1
                if self.consecutive_passes >= 2:
                    self.done = True
                    # Game ends, compute winner
                    player1_cells = np.sum(self.board == 1)
                    player2_cells = np.sum(self.board == -1)
                    if player1_cells > player2_cells:
                        reward = 1 if self.current_player == 1 else -1
                    else:
                        reward = 1 if self.current_player == -1 else -1
                    return self.board.copy(), reward, True, False, self.info
                else:
                    # Switch player
                    self.current_player *= -1
                    return self.board.copy(), 0, False, False, self.info
        else:
            # Attempt to claim a cell
            if self.board[action] != 0:
                # Cell already claimed
                return self.board.copy(), -10, True, False, self.info

            if self.is_first_move():
                # First move, can claim any unclaimed cell
                self.board[action] = self.current_player
                self.consecutive_passes = 0
            else:
                # Subsequent moves, must claim adjacent to own cells
                if self.is_adjacent(action):
                    self.board[action] = self.current_player
                    self.consecutive_passes = 0
                else:
                    # Invalid move (not adjacent)
                    return self.board.copy(), -10, True, False, self.info
            # Check if game over
            if np.all(self.board != 0):
                self.done = True
                # Game ends, compute winner
                player1_cells = np.sum(self.board == 1)
                player2_cells = np.sum(self.board == -1)
                if player1_cells > player2_cells:
                    reward = 1 if self.current_player == 1 else -1
                else:
                    reward = 1 if self.current_player == -1 else -1
                return self.board.copy(), reward, True, False, self.info
            else:
                # Switch player
                self.current_player *= -1
                return self.board.copy(), 0, False, False, self.info

    def render(self):
        grid_str = (
            "   " + "   ".join([str(i + 1) for i in range(self.grid_size)]) + "\n"
        )
        for row in range(self.grid_size):
            row_str = chr(ord("A") + row) + " "
            for col in range(self.grid_size):
                cell_value = self.board[row * self.grid_size + col]
                if cell_value == 1:
                    row_str += " [X]"
                elif cell_value == -1:
                    row_str += " [O]"
                else:
                    row_str += " [ ]"
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        # Returns a list of valid action indices
        valid_actions = []
        # First move
        if self.is_first_move():
            empty_cells = np.where(self.board == 0)[0]
            valid_actions.extend(empty_cells.tolist())
        else:
            # Find all empty cells that are adjacent to current player's claimed cells
            player_cells = np.where(self.board == self.current_player)[0]
            empty_cells = np.where(self.board == 0)[0]
            for cell in empty_cells:
                if self.is_adjacent(cell):
                    valid_actions.append(cell)
        # If no valid actions, can pass
        if len(valid_actions) == 0:
            valid_actions.append(self.num_cells)  # Pass action
        return valid_actions

    def is_first_move(self):
        # Returns True if current player has not claimed any cells yet
        return not np.any(self.board == self.current_player)

    def is_adjacent(self, action):
        # Check if the cell at action is adjacent to any of current player's claimed cells
        player_cells = np.where(self.board == self.current_player)[0]
        row = action // self.grid_size
        col = action % self.grid_size
        for cell in player_cells:
            cell_row = cell // self.grid_size
            cell_col = cell % self.grid_size
            if abs(row - cell_row) + abs(col - cell_col) == 1:
                # Adjacent horizontally or vertically
                return True
        return False
