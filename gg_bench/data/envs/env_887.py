import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            20
        )  # Numbers from 0 to 19 representing 1 to 20
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        # Initialize the number line and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(
            20, dtype=np.int8
        )  # 0: unclaimed, 1: Player 1, -1: Player 2
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board, 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self.board, -10, True, False, {}  # Invalid move

        # Claim the number
        self.board[action] = self.current_player

        # Check if the opponent has any valid moves
        opponent = -self.current_player
        opponent_valid_moves = self._get_valid_moves(opponent)
        if not opponent_valid_moves:
            self.done = True
            return self.board, 1, True, False, {}  # Current player wins

        # Switch players
        self.current_player = opponent
        return self.board, 0, False, False, {}

    def render(self):
        number_line = ""
        for i in range(20):
            num = i + 1
            if self.board[i] == 1:
                number_line += f"[P1-{num}] "
            elif self.board[i] == -1:
                number_line += f"[P2-{num}] "
            else:
                number_line += f"{num} "
        return number_line.strip()

    def valid_moves(self):
        return self._get_valid_moves(self.current_player)

    def _get_valid_moves(self, player):
        claimed_positions = np.where(self.board == player)[0]
        if claimed_positions.size == 0:
            # First move: can claim any unclaimed number
            return [i for i in range(20) if self.board[i] == 0]
        else:
            adjacent_positions = set()
            for pos in claimed_positions:
                if pos > 0 and self.board[pos - 1] == 0:
                    adjacent_positions.add(pos - 1)
                if pos < 19 and self.board[pos + 1] == 0:
                    adjacent_positions.add(pos + 1)
            return list(adjacent_positions)
