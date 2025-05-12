import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The board has 16 cells, actions are 0-15
        self.action_space = spaces.Discrete(16)

        # Observation space representing the 4x4 grid
        # Values: 0 = unclaimed/unblocked, 1 = current player's cell,
        # -1 = opponent's cell, 2 = blocked cell
        self.observation_space = spaces.Box(low=-2, high=2, shape=(16,), dtype=np.int8)

        # Initialize the board
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(16, dtype=np.int8)
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done or self.board[action] != 0:
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Claim the cell
        self.board[action] = self.current_player

        # Block orthogonally adjacent unclaimed cells
        adj_cells = self.get_adjacent_cells(action)
        for adj in adj_cells:
            if self.board[adj] == 0:
                self.board[adj] = 2  # Blocked cell

        # Check if the opponent has any valid moves
        opponent_has_moves = any(self.board[i] == 0 for i in range(16))

        if not opponent_has_moves:
            # Opponent has no moves; current player wins
            self.done = True
            reward = 1  # Reward for winning
        else:
            # Switch to opponent
            self.current_player *= -1
            reward = 0

        return (
            self.board.copy(),
            reward,
            self.done,
            False,
            {},
        )  # Observation, reward, done, truncated, info

    def get_adjacent_cells(self, action):
        row = action // 4
        col = action % 4
        adj_cells = []

        # Up
        if row > 0:
            adj_cells.append((row - 1) * 4 + col)
        # Down
        if row < 3:
            adj_cells.append((row + 1) * 4 + col)
        # Left
        if col > 0:
            adj_cells.append(row * 4 + (col - 1))
        # Right
        if col < 3:
            adj_cells.append(row * 4 + (col + 1))

        return adj_cells

    def has_valid_moves(self):
        return [i for i in range(16) if self.board[i] == 0]

    def render(self):
        board_str = "   1 2 3 4\n"
        for row in range(4):
            row_str = f"{row + 1} ["
            for col in range(4):
                cell_value = self.board[row * 4 + col]
                if cell_value == 1:
                    cell_str = "X"
                elif cell_value == -1:
                    cell_str = "O"
                elif cell_value == 2:
                    cell_str = "#"
                else:
                    cell_str = "-"
                row_str += f"{cell_str} "
            row_str = row_str.strip() + "]\n"
            board_str += row_str
        return board_str

    def valid_moves(self):
        return [i for i in range(16) if self.board[i] == 0]
