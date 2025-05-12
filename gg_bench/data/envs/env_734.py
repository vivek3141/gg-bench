import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(36)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(6, 6), dtype=np.int8)

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((6, 6), dtype=np.int8)
        self.current_player = 1  # Player 1 uses '1', Player 2 uses '2'
        self.done = False
        return self.board, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board, 0, True, False, {}

        # Before processing action, check whether current player has any valid moves
        if not self.has_valid_moves():
            # Current player cannot move, they lose
            self.done = True
            reward = -1  # Current player loses
            return self.board, reward, True, False, {}

        # Map action to grid position
        row = action // 6
        col = action % 6

        # Check if the move is valid
        if self.board[row, col] != 0:
            # Invalid move
            self.done = True
            reward = -10
            return self.board, reward, True, False, {"invalid_move": True}

        # Place the player's block
        self.board[row, col] = self.current_player

        # Block orthogonally adjacent empty cells
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dx, col + dy
            if 0 <= new_row < 6 and 0 <= new_col < 6:
                if self.board[new_row, new_col] == 0:
                    self.board[new_row, new_col] = -1  # Mark as blocked

        # Check if opponent has any valid moves
        opponent = 3 - self.current_player
        if not self.has_valid_moves():
            # Opponent cannot move, current player wins
            self.done = True
            reward = 1
            return self.board, reward, True, False, {}

        # Switch to the next player
        self.current_player = opponent

        # Continue the game
        return self.board, 0, False, False, {}

    def render(self):
        # Create a mapping from numbers to symbols
        symbol_map = {0: ".", 1: "X", 2: "O", -1: "#"}
        board_str = ""
        for row in range(6):
            row_str = " ".join(symbol_map[self.board[row, col]] for col in range(6))
            board_str += row_str + "\n"
        print(board_str)

    def valid_moves(self):
        return [i for i in range(36) if self.board[i // 6, i % 6] == 0]

    def has_valid_moves(self):
        return len(self.valid_moves()) > 0
