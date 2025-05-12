import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The board is 5x5, so there are 25 possible actions (cells)
        self.action_space = spaces.Discrete(25)
        # The observation is the state of the board: 25 cells with values -1, 0, or 1
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(25,), dtype=np.float32
        )

        # Initialize the board and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros(25, dtype=np.float32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.grid, {}  # Return observation and info

    def step(self, action):
        if self.done or action not in self.valid_moves():
            # Invalid move
            self.done = True
            return (
                self.grid.copy(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move
        self.grid[action] = self.current_player

        # Capture adjacent unclaimed cells
        row, col = divmod(action, 5)
        adjacent_indices = []

        # Up
        if row > 0:
            up_index = (row - 1) * 5 + col
            adjacent_indices.append(up_index)
        # Down
        if row < 4:
            down_index = (row + 1) * 5 + col
            adjacent_indices.append(down_index)
        # Left
        if col > 0:
            left_index = row * 5 + (col - 1)
            adjacent_indices.append(left_index)
        # Right
        if col < 4:
            right_index = row * 5 + (col + 1)
            adjacent_indices.append(right_index)

        for idx in adjacent_indices:
            if self.grid[idx] == 0:
                self.grid[idx] = self.current_player

        # Check if the game is over
        if np.all(self.grid != 0):
            # Game over
            self.done = True
            player1_score = np.sum(self.grid == 1)
            player2_score = np.sum(self.grid == -1)
            if (self.current_player == 1 and player1_score > player2_score) or (
                self.current_player == -1 and player2_score > player1_score
            ):
                # Current player wins
                return self.grid.copy(), 1, True, False, {}
            else:
                # Current player loses
                return self.grid.copy(), -1, True, False, {}
        else:
            # Continue the game
            self.current_player *= -1  # Switch player
            return self.grid.copy(), 0, False, False, {}

    def render(self):
        board_str = ""
        symbols = {0: ".", 1: "X", -1: "O"}
        for row in range(5):
            for col in range(5):
                index = row * 5 + col
                board_str += symbols[int(self.grid[index])] + " "
            board_str = board_str.strip() + "\n"
        return board_str.strip()

    def valid_moves(self):
        return [i for i in range(25) if self.grid[i] == 0]
