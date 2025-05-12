import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        # Initialize the game state
        self.board = None  # Will be initialized in reset()
        self.current_player = 1  # Player 1 starts
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(20, dtype=np.int8)
        self.current_player = 1
        self.done = False
        return self.board.copy(), {}  # Return initial observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        if action < 0 or action >= 20 or self.board[action] != 0:
            self.done = True
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Valid move
        self.board[action] = self.current_player

        if np.all(self.board != 0):
            # Game over, determine the winner
            player1_length, player1_sum = self._get_longest_sequence(1)
            player2_length, player2_sum = self._get_longest_sequence(-1)

            if player1_length > player2_length:
                winner = 1
            elif player2_length > player1_length:
                winner = -1
            else:
                if player1_sum > player2_sum:
                    winner = 1
                elif player2_sum > player1_sum:
                    winner = -1
                else:
                    winner = -1  # Player 2 wins tie-breaker

            self.done = True
            if self.current_player == winner:
                return self.board.copy(), 1, True, False, {}
            else:
                return self.board.copy(), 0, True, False, {}
        else:
            # Game continues
            self.current_player *= -1  # Switch player
            return self.board.copy(), 0, False, False, {}

    def render(self):
        line = ""
        for i in range(20):
            if self.board[i] == 1:
                line += " X "
            elif self.board[i] == -1:
                line += " O "
            else:
                line += f" {i+1} "
        return line

    def valid_moves(self):
        return [i for i in range(20) if self.board[i] == 0]

    def _get_longest_sequence(self, player):
        max_len = 0
        max_sum = 0
        current_len = 0
        current_sum = 0

        for i in range(20):
            if self.board[i] == player:
                current_len += 1
                current_sum += i + 1  # Number line positions are 1-based
                if current_len > max_len:
                    max_len = current_len
                    max_sum = current_sum
                elif current_len == max_len:
                    if current_sum > max_sum:
                        max_sum = current_sum
            else:
                current_len = 0
                current_sum = 0

        return max_len, max_sum
