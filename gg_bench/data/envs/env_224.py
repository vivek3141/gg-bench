import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Actions correspond to numbers 1-9
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(9,), dtype=np.int8
        )  # -1: picked by Player 2, 0: available, 1: picked by Player 1

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(
            9, dtype=np.int8
        )  # Initialize number pool 1-9 as available
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), -10, True, False, {}

        if self.board[action] != 0:
            # Invalid move (number already picked)
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Valid move
        self.board[action] = self.current_player

        # Check for win condition
        player_numbers = (
            np.where(self.board == self.current_player)[0] + 1
        )  # Numbers picked by current player
        if len(player_numbers) >= 3:
            # Check all combinations of three numbers
            for combo in itertools.combinations(player_numbers, 3):
                if self.is_arithmetic_sequence(combo):
                    self.done = True
                    return self.board.copy(), 1, True, False, {}

        # Check for endgame condition (all numbers picked)
        if np.all(self.board != 0):
            # No sequence formed, last player to pick loses
            self.done = True
            return self.board.copy(), -1, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        return self.board.copy(), 0, False, False, {}

    def render(self):
        status = "Number Pool Status:\n"
        for i in range(9):
            num = i + 1
            if self.board[i] == 0:
                status += f"{num}: Available\n"
            elif self.board[i] == 1:
                status += f"{num}: Picked by Player 1\n"
            elif self.board[i] == -1:
                status += f"{num}: Picked by Player 2\n"
        return status

    def valid_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    @staticmethod
    def is_arithmetic_sequence(combo):
        seq = sorted(combo)
        return seq[1] - seq[0] == seq[2] - seq[1]
