import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is discrete [0, 24] representing positions in the 5x5 grid
        self.action_space = spaces.Discrete(25)
        # The observation is the state of the grid, 5x5 grid with values -1, 0, 1
        # We use dtype int8 to represent the small integers
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        # Initialize the board
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # 1 for Player 1 ('X'), -1 for Player 2 ('O')
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        row = action // 5
        col = action % 5

        # Check for valid move
        if not (0 <= row < 5 and 0 <= col < 5) or self.board[row, col] != 0:
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Place the current player's token
        self.board[row, col] = self.current_player

        # Check for win
        if self.check_win():
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch player
        self.current_player *= -1

        return self.board.copy(), 0, False, False, {}

    def render(self):
        # Return a visual representation of the board
        board_str = ""
        for row in self.board:
            row_str = ""
            for cell in row:
                if cell == 1:
                    row_str += " X "
                elif cell == -1:
                    row_str += " O "
                else:
                    row_str += " . "
            board_str += row_str + "\n"
        return board_str

    def valid_moves(self):
        return [i for i in range(25) if self.board[i // 5, i % 5] == 0]

    def check_win(self):
        from collections import deque

        visited = np.zeros((5, 5), dtype=bool)

        if self.current_player == 1:
            # Player 1: check for path from top to bottom
            queue = deque()
            # Enqueue all positions in the top row that have current_player's token
            for col in range(5):
                if self.board[0, col] == self.current_player:
                    queue.append((0, col))
                    visited[0, col] = True

            while queue:
                row, col = queue.popleft()
                if row == 4:
                    return True  # Reached bottom row
                # Explore neighbors
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < 5 and 0 <= new_col < 5:
                            if (
                                not visited[new_row, new_col]
                                and self.board[new_row, new_col] == self.current_player
                            ):
                                visited[new_row, new_col] = True
                                queue.append((new_row, new_col))
        else:
            # Player 2: check for path from left to right
            queue = deque()
            # Enqueue all positions in the left column that have current_player's token
            for row in range(5):
                if self.board[row, 0] == self.current_player:
                    queue.append((row, 0))
                    visited[row, 0] = True

            while queue:
                row, col = queue.popleft()
                if col == 4:
                    return True  # Reached rightmost column
                # Explore neighbors
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < 5 and 0 <= new_col < 5:
                            if (
                                not visited[new_row, new_col]
                                and self.board[new_row, new_col] == self.current_player
                            ):
                                visited[new_row, new_col] = True
                                queue.append((new_row, new_col))
        return False
