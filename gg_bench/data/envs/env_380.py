import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the board and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # 1 for 'X', -1 for 'O'
        self.done = False
        self.last_capture_player = 0  # Tracks the last player who made a capture
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        terminated = False
        truncated = False

        if self.done or self.board[action] != 0:
            # Invalid move
            terminated = True
            reward = -10
            return self.board.copy(), reward, terminated, truncated, {}

        # Place the current player's symbol
        self.board[action] = self.current_player
        row, col = divmod(action, 3)
        captures_made = self._apply_captures(row, col)

        if captures_made:
            self.last_capture_player = self.current_player

        # Check for a win
        player_squares = np.sum(self.board == self.current_player)
        if player_squares > 4:
            terminated = True
            reward = 1
            self.done = True
            return self.board.copy(), reward, terminated, truncated, {}

        # Check if the board is full
        if np.all(self.board != 0):
            terminated = True
            self.done = True
            # Determine winner based on majority
            player1_squares = np.sum(self.board == 1)
            player2_squares = np.sum(self.board == -1)
            if player1_squares > player2_squares:
                winner = 1
            elif player2_squares > player1_squares:
                winner = -1
            else:
                winner = self.last_capture_player

            if winner == self.current_player:
                reward = 1
            else:
                reward = 0
            return self.board.copy(), reward, terminated, truncated, {}

        # Continue the game
        reward = 0
        self.current_player *= -1
        return self.board.copy(), reward, terminated, truncated, {}

    def render(self):
        symbols = {1: "X", -1: "O", 0: " "}
        board_str = "\n-------------\n"
        for i in range(3):
            board_str += "|"
            for j in range(3):
                idx = i * 3 + j
                board_str += f" {symbols[self.board[idx]]} |"
            board_str += "\n-------------\n"
        return board_str

    def valid_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def _apply_captures(self, row, col):
        captures_made = False
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dr, dc in directions:
            captures = []
            r, c = row + dr, col + dc
            while 0 <= r < 3 and 0 <= c < 3:
                idx = r * 3 + c
                if self.board[idx] == -self.current_player:
                    captures.append(idx)
                elif self.board[idx] == self.current_player:
                    if captures:
                        # Capture the opponent's pieces
                        for capture_idx in captures:
                            self.board[capture_idx] = self.current_player
                        captures_made = True
                    break
                else:
                    break
                r += dr
                c += dc
        return captures_made
