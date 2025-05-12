import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(
            21
        )  # Actions correspond to positions 0-20 (numbers 1-21)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(21,), dtype=np.int8)

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(
            21, dtype=np.int8
        )  # 0: Unclaimed, 1: Player 1, -1: Player 2, 2: Blocked
        self.current_player = 1  # Player 1 starts
        self.terminated = False
        self.truncated = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.terminated:
            return self.board.copy(), 0, True, False, {}

        # Check if the action is within the valid range
        if action < 0 or action >= 21:
            self.terminated = True
            return self.board.copy(), -10, True, False, {}

        # Check if the selected number is unclaimed and unblocked
        if self.board[action] != 0:
            self.terminated = True
            return self.board.copy(), -10, True, False, {}

        # Valid move: claim the number
        self.board[action] = self.current_player

        # Block adjacent unclaimed numbers
        if action > 0 and self.board[action - 1] == 0:
            self.board[action - 1] = 2  # Mark as blocked
        if action < 20 and self.board[action + 1] == 0:
            self.board[action + 1] = 2  # Mark as blocked

        # Check if the opponent has any valid moves
        opponent = -self.current_player
        opponent_valid_moves = [i for i in range(21) if self.board[i] == 0]

        if not opponent_valid_moves:
            # Opponent cannot move; current player wins
            self.terminated = True
            return self.board.copy(), 1, True, False, {}
        else:
            # Switch to the opponent
            self.current_player = opponent
            return self.board.copy(), 0, False, False, {}

    def render(self):
        board_str = ""
        for i in range(21):
            val = self.board[i]
            if val == 0:
                board_str += f"{i + 1} "  # Unclaimed number
            elif val == 1:
                board_str += "X "  # Claimed by Player 1
            elif val == -1:
                board_str += "O "  # Claimed by Player 2
            elif val == 2:
                board_str += "- "  # Blocked
        return board_str.strip()

    def valid_moves(self):
        # Returns indices of unclaimed and unblocked numbers
        return [i for i in range(21) if self.board[i] == 0]
