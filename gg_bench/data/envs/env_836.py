import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Actions correspond to numbers 1 to 9
        self.observation_space = spaces.Box(low=-1, high=2, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 0: Unclaimed, not forbidden
        # -1: Forbidden
        # 1: Claimed by Player 1
        # 2: Claimed by Player 2
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        if action < 0 or action >= 9:
            return self.board.copy(), -10, True, False, {}  # Invalid action

        if self.board[action] != 0:
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Claim the number
        self.board[action] = self.current_player

        # Mark adjacent numbers as forbidden
        adjacent_indices = []
        if action > 0:
            adjacent_indices.append(action - 1)
        if action < 8:
            adjacent_indices.append(action + 1)

        for idx in adjacent_indices:
            if self.board[idx] == 0:
                self.board[idx] = -1  # Mark as forbidden

        # Check if opponent has any valid moves
        self.current_player = 3 - self.current_player  # Switch player: 1 <-> 2
        if not self.valid_moves():
            self.done = True
            reward = 1  # Current player wins
            return self.board.copy(), reward, True, False, {}

        # Switch back to current player for next move
        self.current_player = 3 - self.current_player
        return self.board.copy(), 0, False, False, {}

    def render(self):
        board_str = "Number Line: "
        symbols = {0: "{}", -1: "[{}]", 1: "P1", 2: "P2"}
        for idx, val in enumerate(self.board):
            number = idx + 1
            if val == 0:
                board_str += f"{number} "
            elif val == -1:
                board_str += f"[{number}] "
            elif val == 1:
                board_str += "P1 "
            elif val == 2:
                board_str += "P2 "
        return board_str.strip()

    def valid_moves(self):
        valid_moves = []
        for idx, val in enumerate(self.board):
            if val == 0:
                valid_moves.append(idx)
        return valid_moves
