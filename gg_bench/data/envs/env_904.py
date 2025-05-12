import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=0, high=2, shape=(11,), dtype=np.int32)

        # Initialize the board and current player
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(
            11, dtype=np.int32
        )  # 0: Available, 1: Isolated, 2: Removed
        self.current_player = 1
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        if action not in self.valid_moves():
            # Invalid move
            reward = -10
            self.done = True
            return self.board.copy(), reward, True, False, {}

        # Perform the action
        self.board[action] = 2  # Mark as removed

        # Isolate adjacent numbers
        if action > 0 and self.board[action - 1] == 0:
            self.board[action - 1] = 1  # Mark as isolated
        if action < 10 and self.board[action + 1] == 0:
            self.board[action + 1] = 1  # Mark as isolated

        # Check for win conditions
        if np.all(self.board != 0):
            # Current player removed the last available number
            reward = 1
            self.done = True
            return self.board.copy(), reward, True, False, {}
        elif len(self.valid_moves()) == 0:
            # Opponent has no valid moves
            reward = 1
            self.done = True
            return self.board.copy(), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            self.current_player *= -1  # Switch to other player
            return self.board.copy(), reward, False, False, {}

    def render(self):
        sequence = ""
        for i in range(11):
            if self.board[i] == 0:
                sequence += f"{i+1} "
            elif self.board[i] == 1:
                sequence += f"[{i+1}] "
            elif self.board[i] == 2:
                sequence += f"({i+1}) "
        return sequence.strip()

    def valid_moves(self):
        return [i for i in range(11) if self.board[i] == 0]
