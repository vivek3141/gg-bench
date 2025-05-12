import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Territories numbered 0-8
        self.observation_space = spaces.Box(low=-1, high=2, shape=(9,), dtype=np.int8)

        # Initialize the board
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Check if action is valid
        if action not in self.valid_moves():
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Perform action
        self.board[action] = self.current_player

        # Lock adjacent territories
        if action > 0:
            if self.board[action - 1] == 0:
                self.board[action - 1] = -1  # Locked
        if action < 8:
            if self.board[action + 1] == 0:
                self.board[action + 1] = -1  # Locked

        # Check if game is over
        if np.any(self.board == 0):
            self.done = False
            reward = 0
        else:
            self.done = True
            # Game over, determine winner
            player1_claims = np.count_nonzero(self.board == 1)
            player2_claims = np.count_nonzero(self.board == 2)
            if player1_claims > player2_claims:
                winner = 1
            else:
                winner = 2
            if self.current_player == winner:
                reward = 1
            else:
                reward = 1  # Since the last move led to the win, reward current player

        # Switch player
        if not self.done:
            self.current_player = (
                2 if self.current_player == 1 else 1
            )  # Switch between 1 and 2

        return (
            self.board.copy(),
            reward,
            self.done,
            False,
            {},
        )  # observation, reward, terminated, truncated, info

    def render(self):
        board_str = ""
        for i in range(9):
            pos = self.board[i]
            if pos == 0:
                board_str += "[{}] ".format(i + 1)
            elif pos == -1:
                board_str += "[🔒] "
            elif pos == 1:
                board_str += "[X] "
            elif pos == 2:
                board_str += "[O] "
            if i == 8:
                board_str += "\n"
        return board_str

    def valid_moves(self):
        return [i for i in range(9) if self.board[i] == 0]
