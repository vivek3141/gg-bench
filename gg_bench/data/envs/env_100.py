import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 25 possible actions (cells in a 5x5 grid)
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(25,), dtype=np.int32)

        self.grid_size = 5  # 5x5 grid
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(25, dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}  # Game is over

        if action < 0 or action >= 25 or self.board[action] != 0:
            # Invalid action: action out of bounds or cell already claimed
            self.done = True
            return self.board.copy(), -10, True, False, {}

        if not self.is_valid_move(action, self.current_player):
            # Invalid move according to game rules
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Apply the action
        self.board[action] = self.current_player

        # Check if opponent has valid moves
        self.current_player *= -1  # Switch to opponent
        opponent_valid_moves = self.valid_moves()

        if len(opponent_valid_moves) == 0:
            # Opponent cannot move, current player wins
            self.done = True
            reward = 1  # Reward to the current player
            self.current_player *= -1  # Switch back to current player
            return self.board.copy(), reward, True, False, {}

        # Opponent can move, game continues
        # Keep current_player as opponent for the next turn
        # No immediate reward
        return self.board.copy(), 0, False, False, {}

    def render(self):
        board_str = "Current board state:\n"
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                cell = self.board[i * self.grid_size + j]
                if cell == 1:
                    row_str += " X "
                elif cell == -1:
                    row_str += " O "
                else:
                    row_str += " . "
            board_str += row_str + "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        for action in range(self.action_space.n):
            if self.board[action] == 0 and self.is_valid_move(
                action, self.current_player
            ):
                valid_actions.append(action)
        return valid_actions

    def is_valid_move(self, action, player):
        if self.board[action] != 0:
            return False  # Cell already claimed

        # First move for the player, can claim any unclaimed cell
        if not (self.board == player).any():
            return True

        # Get position of the action
        row = action // self.grid_size
        col = action % self.grid_size

        # Get positions of player's claimed cells
        player_cells = np.where(self.board == player)[0]
        for cell in player_cells:
            cell_row = cell // self.grid_size
            cell_col = cell % self.grid_size

            # Check adjacency
            if abs(row - cell_row) <= 1 and abs(col - cell_col) <= 1:
                return False  # Adjacent to player's cell

        return True  # Valid move
