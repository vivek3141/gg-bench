import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is the 25 cells in the grid (0 to 24)
        self.action_space = spaces.Discrete(25)
        # The observation space is a 5x5 grid with values:
        #  0: Empty and Unlocked
        #  1: Marked by Player A
        #  2: Marked by Player B
        # -1: Locked
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=np.int8)

        # Initialize the board
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # Player A is 1, Player B is 2
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}
        row, col = divmod(action, 5)

        # Check if the move is valid
        if self.board[row, col] != 0:
            # Invalid move
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Place the marker
        self.board[row, col] = self.current_player

        # Lock the selected cell and orthogonally adjacent cells
        neighbors = [
            (row, col),
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]
        for r, c in neighbors:
            if 0 <= r < 5 and 0 <= c < 5:
                if self.board[r, c] == 0:
                    self.board[r, c] = -1  # Lock the cell

        # Check if the opponent has any valid moves
        valid_moves_next_player = [
            i for i in range(25) if self.board[i // 5, i % 5] == 0
        ]
        if not valid_moves_next_player:
            # Opponent has no valid moves, current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch to the next player
        self.current_player = 1 if self.current_player == 2 else 2

        # Game continues
        return self.board.copy(), -10, False, False, {}

    def render(self):
        board_str = ""
        board_str += "   " + "  ".join(str(c + 1) for c in range(5)) + "\n"
        for r in range(5):
            board_str += str(r + 1) + " "
            for c in range(5):
                val = self.board[r, c]
                if val == 0:
                    board_str += "[ ]"
                elif val == -1:
                    board_str += "[L]"
                elif val == 1:
                    board_str += "[A]"
                elif val == 2:
                    board_str += "[B]"
            board_str += "\n"
        return board_str

    def valid_moves(self):
        return [i for i in range(25) if self.board[i // 5, i % 5] == 0]
