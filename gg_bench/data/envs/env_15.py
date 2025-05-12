import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space represents the destination squares (0-15) on the 4x4 grid
        self.action_space = spaces.Discrete(16)
        # The observation space is a 4x4 grid with values: 1 (Player 1's Knight), -1 (Player 2's Knight), 0 (empty)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4, 4), dtype=np.int8)

        # Initialize the board
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((4, 4), dtype=np.int8)
        self.board[0, 0] = 1  # Player 1's Knight at A1
        self.board[3, 3] = -1  # Player 2's Knight at D4
        self.current_player = 1
        self.done = False
        return np.copy(self.board), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return np.copy(self.board), 0, True, False, {}

        # Get valid moves for the current player
        valid_moves = self.valid_moves()
        if action not in valid_moves:
            self.done = True
            return np.copy(self.board), -10, True, False, {}

        # Get current position of the current player's Knight
        current_pos = np.argwhere(self.board == self.current_player)
        if current_pos.size == 0:
            self.done = True
            return np.copy(self.board), -10, True, False, {}
        current_x, current_y = current_pos[0]

        # Move the Knight
        new_x = action // 4
        new_y = action % 4

        # Check if capturing opponent's Knight
        if self.board[new_x, new_y] == -self.current_player:
            # Capture opponent's Knight
            self.board[current_x, current_y] = 0
            self.board[new_x, new_y] = self.current_player
            self.done = True
            return np.copy(self.board), 1, True, False, {}

        # Move to empty cell
        self.board[current_x, current_y] = 0
        self.board[new_x, new_y] = self.current_player

        # Check if opponent has any valid moves
        # Temporarily switch player to check opponent's moves
        self.current_player *= -1
        opponent_valid_moves = self.valid_moves()
        if not opponent_valid_moves:
            # Opponent has no valid moves; current player wins
            self.current_player *= -1  # Switch back to current player
            self.done = True
            return np.copy(self.board), 1, True, False, {}

        # Switch to opponent's turn
        self.current_player *= -1
        return np.copy(self.board), 0, False, False, {}

    def valid_moves(self):
        # Get current position of the current player's Knight
        current_pos = np.argwhere(self.board == self.current_player)
        if current_pos.size == 0:
            return []

        x, y = current_pos[0]
        # Possible Knight moves (L-shaped moves)
        moves = [
            (x + 2, y + 1),
            (x + 1, y + 2),
            (x - 1, y + 2),
            (x - 2, y + 1),
            (x - 2, y - 1),
            (x - 1, y - 2),
            (x + 1, y - 2),
            (x + 2, y - 1),
        ]

        valid_moves = []
        for new_x, new_y in moves:
            if 0 <= new_x < 4 and 0 <= new_y < 4:
                if self.board[new_x, new_y] != self.current_player:
                    index = new_x * 4 + new_y
                    valid_moves.append(index)
        return valid_moves

    def render(self):
        board_str = "    A   B   C   D\n" "  +---+---+---+---+\n"
        for i in range(4):
            row_str = f"{i + 1} |"
            for j in range(4):
                cell = self.board[i, j]
                if cell == 1:
                    row_str += "N1 |"
                elif cell == -1:
                    row_str += "N2 |"
                else:
                    row_str += " . |"
            row_str += "\n  +---+---+---+---+\n"
            board_str += row_str
        return board_str

    def valid_moves(self):
        # Get current position of current player's Knight
        current_pos = np.argwhere(self.board == self.current_player)
        if current_pos.size == 0:
            return []

        x, y = current_pos[0]
        # Possible Knight moves (L-shaped moves)
        moves = [
            (x + 2, y + 1),
            (x + 1, y + 2),
            (x - 1, y + 2),
            (x - 2, y + 1),
            (x - 2, y - 1),
            (x - 1, y - 2),
            (x + 1, y - 2),
            (x + 2, y - 1),
        ]

        valid_moves = []
        for new_x, new_y in moves:
            if 0 <= new_x < 4 and 0 <= new_y < 4:
                if self.board[new_x, new_y] != self.current_player:
                    index = new_x * 4 + new_y
                    valid_moves.append(index)
        return valid_moves
