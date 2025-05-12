import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(25)  # Grid of 5x5 cells
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(25,), dtype=np.float32
        )

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(25, dtype=np.float32)  # 5x5 grid flattened
        self.current_player = 1  # Player 1 starts (represented by 1), Player 2 is -1
        self.last_move_player = None  # Tracks who made the last valid move
        self.done = False  # Game over flag
        self.passed = {1: False, -1: False}  # Tracks if players have passed
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Check if the current player has any valid moves
        valid_moves = self.valid_moves()
        if not valid_moves:
            # Current player must pass
            self.passed[self.current_player] = True

            if self.passed[1] and self.passed[-1]:
                # Both players have passed, game is over
                self.done = True
                if self.last_move_player == self.current_player:
                    # Current player made the last valid move and wins
                    reward = 1
                else:
                    # Current player loses
                    reward = 0
                return self.board.copy(), reward, True, False, {}

            else:
                # Switch to the next player
                self.current_player *= -1
                return self.board.copy(), 0, False, False, {}

        else:
            # Current player has valid moves
            self.passed[self.current_player] = False

            # Check if the action is valid
            if action not in valid_moves:
                # Invalid move
                self.done = True
                return self.board.copy(), -10, True, False, {}

            # Place the token on the board
            self.board[action] = self.current_player

            # Update the last move player
            self.last_move_player = self.current_player

            # Reward for making a valid move is -10
            reward = -10

            # Switch to the next player
            self.current_player *= -1

            return self.board.copy(), reward, False, False, {}

    def render(self):
        board_str = ""
        for i in range(5):
            row_str = ""
            for j in range(5):
                val = self.board[i * 5 + j]
                if val == 1:
                    row_str += "X "
                elif val == -1:
                    row_str += "O "
                else:
                    row_str += "_ "
            board_str += row_str.strip()
            board_str += "\n"
        return board_str

    def valid_moves(self):
        current_player = self.current_player
        valid_moves = []
        for idx in range(25):
            if self.board[idx] != 0:
                continue
            # Check if cell is adjacent to any of the current player's tokens
            row = idx // 5
            col = idx % 5
            is_adjacent_to_own = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r = row + dr
                    c = col + dc
                    if 0 <= r < 5 and 0 <= c < 5:
                        neighbor_idx = r * 5 + c
                        if self.board[neighbor_idx] == current_player:
                            is_adjacent_to_own = True
                            break
                if is_adjacent_to_own:
                    break
            if not is_adjacent_to_own:
                valid_moves.append(idx)
        return valid_moves
