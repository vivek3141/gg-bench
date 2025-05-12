import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.grid_size = 4
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.grid_size, self.grid_size), dtype=np.uint8
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.current_player = 1  # 1 for Player 1 ('X'), 2 for Player 2 ('O')
        self.done = False
        return self.grid.copy(), {}

    def step(self, action):
        if self.done:
            return self.grid.copy(), -10, True, False, {}

        row = action // self.grid_size
        col = action % self.grid_size

        if not self.is_valid_move(row, col, self.current_player):
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        self.grid[row, col] = self.current_player

        opponent = 2 if self.current_player == 1 else 1
        if not self.has_valid_moves(opponent):
            self.done = True
            return self.grid.copy(), 1, True, False, {}

        self.current_player = opponent
        return self.grid.copy(), 0, False, False, {}

    def render(self):
        symbol_map = {0: ".", 1: "X", 2: "O"}
        board_str = ""
        for row in self.grid:
            board_str += " ".join(symbol_map[cell] for cell in row) + "\n"
        return board_str

    def valid_moves(self):
        valid_moves = []
        for action in range(self.grid_size * self.grid_size):
            row = action // self.grid_size
            col = action % self.grid_size
            if self.is_valid_move(row, col, self.current_player):
                valid_moves.append(action)
        return valid_moves

    def is_valid_move(self, row, col, player):
        if self.grid[row, col] != 0:
            return False

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = row + dr, col + dc
                if (
                    (dr != 0 or dc != 0)
                    and 0 <= r < self.grid_size
                    and 0 <= c < self.grid_size
                ):
                    if self.grid[r, c] == player:
                        return False
        return True

    def has_valid_moves(self, player):
        for action in range(self.grid_size * self.grid_size):
            row = action // self.grid_size
            col = action % self.grid_size
            if self.is_valid_move(row, col, player):
                return True
        return False
