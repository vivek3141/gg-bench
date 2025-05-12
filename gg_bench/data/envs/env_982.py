import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(9), actions 0 to 8 correspond to numbers 1 to 9
        self.action_space = spaces.Discrete(9)
        # The observation space is a Box of length 9, with values -1.0, 0.0, or 1.0
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.float32)
        self.current_player = 1  # Player 1 starts
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0

        if self.board[action] != 0:
            # Invalid move: number already taken
            reward = -10
            terminated = True
            return self.board.copy(), reward, terminated, truncated, {}

        # Valid move
        self.board[action] = self.current_player

        # Check if current player has formed an arithmetic sequence
        selected_numbers = [
            i + 1 for i in range(9) if self.board[i] == self.current_player
        ]
        if len(selected_numbers) >= 3:
            # Check all combinations of 3 numbers
            for combo in combinations(selected_numbers, 3):
                if combo[1] - combo[0] == combo[2] - combo[1]:
                    # Arithmetic sequence found
                    reward = 1
                    terminated = True
                    return self.board.copy(), reward, terminated, truncated, {}

        if np.all(self.board != 0):
            # All numbers have been selected and no player has won
            # The last player to pick a number loses (current player)
            reward = -1
            terminated = True
            return self.board.copy(), reward, terminated, truncated, {}

        # Switch to the other player
        self.current_player *= -1
        return self.board.copy(), reward, terminated, truncated, {}

    def render(self):
        board_str = "Numbers selected:\n"
        for i in range(9):
            if self.board[i] == 1:
                board_str += f"Player 1 picked {i + 1}\n"
            elif self.board[i] == -1:
                board_str += f"Player 2 picked {i + 1}\n"
            else:
                board_str += f"{i + 1} is available\n"
        return board_str

    def valid_moves(self):
        return [i for i in range(9) if self.board[i] == 0]
