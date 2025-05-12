import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )

        # Initialize the board and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(16, dtype=np.float32)
        self.current_player = 1  # 1 for Player 1 ('X'), -1 for Player 2 ('O')
        self.done = False
        return self.board.copy(), {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Place marker
        self.board[action] = self.current_player

        # Check if opponent has any valid moves
        opponent = -self.current_player  # Opponent player
        opponent_valid_moves = self._get_valid_moves_for_player(opponent)

        if len(opponent_valid_moves) == 0:
            # Opponent has no valid moves, current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}
        else:
            # Game continues
            self.current_player = opponent  # Switch to opponent
            return self.board.copy(), 0, False, False, {}

    def render(self):
        grid_str = "    1   2   3   4\n"
        grid_str += "  +---+---+---+---+\n"
        for row in range(4):
            grid_str += "{} |".format(row + 1)
            for col in range(4):
                idx = row * 4 + col
                if self.board[idx] == 1:
                    grid_str += " X |"
                elif self.board[idx] == -1:
                    grid_str += " O |"
                else:
                    grid_str += "   |"
            grid_str += "\n  +---+---+---+---+\n"
        return grid_str

    def valid_moves(self):
        return self._get_valid_moves_for_player(self.current_player)

    def _get_valid_moves_for_player(self, player):
        valid_moves = []
        opponent = -player
        for idx in range(16):
            if self.board[idx] != 0:
                continue  # Cell is not empty
            row = idx // 4
            col = idx % 4
            # Check adjacent cells for opponent's markers
            has_adjacent_opponent = False
            # Up
            if row > 0:
                up_idx = (row - 1) * 4 + col
                if self.board[up_idx] == opponent:
                    has_adjacent_opponent = True
            # Down
            if row < 3:
                down_idx = (row + 1) * 4 + col
                if self.board[down_idx] == opponent:
                    has_adjacent_opponent = True
            # Left
            if col > 0:
                left_idx = row * 4 + (col - 1)
                if self.board[left_idx] == opponent:
                    has_adjacent_opponent = True
            # Right
            if col < 3:
                right_idx = row * 4 + (col + 1)
                if self.board[right_idx] == opponent:
                    has_adjacent_opponent = True
            if not has_adjacent_opponent:
                valid_moves.append(idx)
        return valid_moves
