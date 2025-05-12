import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(25)  # 5x5 grid has 25 positions
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(5, 5), dtype=np.int8
        )  # -1 for 'O', 0 for empty, 1 for 'X'

        # Define the 'L' pattern masks for Player 1 ('X')
        self.L_patterns = self.generate_L_patterns()
        # Define the 'T' pattern masks for Player 2 ('O')
        self.T_patterns = self.generate_T_patterns()

        # Initialize the board
        self.reset()

    def generate_L_patterns(self):
        # Base 'L' pattern for Player 1
        pattern = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.int8)
        patterns = []
        for k in range(4):  # Rotate pattern
            rotated_pattern = np.rot90(pattern, k)
            patterns.append(rotated_pattern)
            patterns.append(np.fliplr(rotated_pattern))  # Reflect pattern
        return patterns

    def generate_T_patterns(self):
        # Base 'T' pattern for Player 2
        pattern = np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=np.int8)
        patterns = []
        for k in range(4):  # Rotate pattern
            rotated_pattern = np.rot90(pattern, k)
            patterns.append(rotated_pattern)
            patterns.append(np.fliplr(rotated_pattern))  # Reflect pattern
        return patterns

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # 1 for 'X', -1 for 'O'
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        row = action // 5
        col = action % 5

        if self.done:
            return self.board.copy(), 0, True, False, {}

        if self.board[row, col] != 0:
            # Invalid move
            self.done = True
            return self.board.copy(), -10, True, False, {}

        self.board[row, col] = self.current_player

        # Check for win
        if self.check_win():
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Check for draw (unlikely in this game)
        if np.all(self.board != 0):
            self.done = True
            return self.board.copy(), 0, True, False, {}

        # Switch player
        self.current_player *= -1
        return self.board.copy(), 0, False, False, {}

    def check_win(self):
        player = self.current_player
        patterns = self.L_patterns if player == 1 else self.T_patterns
        player_board = (self.board == player).astype(np.int8)
        for pattern in patterns:
            for row in range(self.board.shape[0] - 2):
                for col in range(self.board.shape[1] - 2):
                    sub_board = player_board[row : row + 3, col : col + 3]
                    if sub_board.shape != (3, 3):
                        continue
                    if np.all(sub_board[pattern == 1] == 1):
                        return True
        return False

    def render(self):
        board_str = "   1   2   3   4   5\n"
        for i in range(5):
            row_str = f"{i + 1} "
            for j in range(5):
                cell = self.board[i, j]
                if cell == 1:
                    row_str += " X "
                elif cell == -1:
                    row_str += " O "
                else:
                    row_str += " . "
            board_str += row_str.rstrip() + "\n"
        return board_str

    def valid_moves(self):
        return [i for i in range(25) if self.board[i // 5, i % 5] == 0]
