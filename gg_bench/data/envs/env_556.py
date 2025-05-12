import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action_space is Discrete(13), for positions 0 to 12 (positions 1 to 13)
        self.action_space = spaces.Discrete(13)
        # The observation_space is a Box space with values in [-1, 2]:
        # -1: claimed by Player 2 (O)
        #  0: unclaimed and unblocked
        #  1: claimed by Player 1 (X)
        #  2: blocked
        self.observation_space = spaces.Box(low=-1, high=2, shape=(13,), dtype=np.int32)

        # Initialize the board
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(13, dtype=np.int32)
        self.current_player = 1  # Player 1 starts (marker 1), Player 2 is -1
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def valid_moves(self):
        if self.done:
            return []
        # Valid moves are positions where the board is 0 (unclaimed and unblocked)
        return [i for i in range(13) if self.board[i] == 0]

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Check if current player has valid moves
        valid_actions = self.valid_moves()
        if len(valid_actions) == 0:
            # Current player has no valid moves, current player loses
            self.done = True
            reward = -1  # Current player loses
            return self.board.copy(), reward, True, False, {}

        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return self.board.copy(), reward, True, False, {}

        # Valid move
        # Claim the position
        self.board[action] = self.current_player

        # Block adjacent positions if they are unclaimed and unblocked
        for pos in [action - 1, action + 1]:
            if 0 <= pos < 13 and self.board[pos] == 0:
                self.board[pos] = 2  # Blocked

        # Check if opponent has any valid moves
        opponent = -self.current_player  # Opponent's marker
        # Switch to opponent
        self.current_player = opponent
        opponent_valid_moves = self.valid_moves()

        if len(opponent_valid_moves) == 0:
            # Opponent has no valid moves, current player (previous player) wins
            self.done = True
            reward = 1  # The player who made the move wins
            return self.board.copy(), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            return self.board.copy(), reward, False, False, {}

    def render(self):
        position_line = "Positions: "
        markers_line = "Markers:   "
        for i in range(13):
            position_line += f"{i+1} "
            if self.board[i] == 0:
                markers_line += f"{i+1} "
            elif self.board[i] == 1:
                markers_line += "X "
            elif self.board[i] == -1:
                markers_line += "O "
            elif self.board[i] == 2:
                markers_line += "- "
        board_str = position_line + "\n" + markers_line
        return board_str
