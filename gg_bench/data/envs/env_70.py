import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(7)  # Coins 1 to 7
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8)

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        # Reset the environment to the initial state
        super().reset(seed=seed)
        self.board = np.ones(
            7, dtype=np.int8
        )  # All coins are heads up (1 represents heads up)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        if action < 0 or action >= 7 or self.board[action] == 0:
            # Invalid move: selected coin is tails up or out of range
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {},
            )  # End the game with a penalty

        # Flip the selected coin and immediate right adjacent heads-up coins
        idx = action
        while idx < 7 and self.board[idx] == 1:
            self.board[idx] = 0  # Flip to tails up (0 represents tails up)
            idx += 1

        # Check if the current player wins
        if np.all(self.board == 0):
            self.done = True
            return (
                self.board.copy(),
                1,
                True,
                False,
                {},
            )  # Current player wins with a reward of 1

        # Switch to the next player
        self.current_player *= -1
        return (
            self.board.copy(),
            -10,
            False,
            False,
            {},
        )  # Valid move with a penalty of -10

    def render(self):
        # Generate a string representation of the game board
        board_str = ""
        for idx, coin in enumerate(self.board):
            state = "H" if coin == 1 else "T"
            board_str += f"[{idx + 1}:{state}]"
        return board_str

    def valid_moves(self):
        # Return a list of valid moves (indices of heads-up coins)
        return [idx for idx, coin in enumerate(self.board) if coin == 1]
