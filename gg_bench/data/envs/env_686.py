import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is a Discrete space with 16 possible actions (4x4 grid)
        self.action_space = spaces.Discrete(16)

        # The observation is a flattened 4x4 grid with values:
        # 0 for empty, 1 for Player 1's marker, -1 for Player 2's marker
        self.observation_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the grid to empty
        self.grid = np.zeros(16, dtype=np.int8)

        # Player 1 uses 'X' (represented by 1), Player 2 uses 'O' (represented by -1)
        self.current_player = 1  # Player 1 starts

        # No last move at the beginning
        self.last_move = None

        # Game is not over
        self.done = False

        return self.grid.copy(), {}

    def step(self, action):
        if self.done:
            # If the game is over, return the current state
            return self.grid.copy(), 0, True, False, {}

        if self.grid[action] != 0:
            # Invalid move: Cell is already occupied
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Convert action to (row, col)
        row = action // 4
        col = action % 4

        if self.last_move is not None:
            last_row, last_col = self.last_move
            # Check adjacency
            if not (
                (row == last_row and abs(col - last_col) == 1)
                or (col == last_col and abs(row - last_row) == 1)
            ):
                # Invalid move: Not adjacent to the last move
                self.done = True
                return self.grid.copy(), -10, True, False, {}

        # Place the marker
        self.grid[action] = self.current_player
        self.last_move = (row, col)

        # Check if the opponent has any valid moves
        opponent_valid_moves = self.get_valid_moves(self.last_move, self.grid)
        if not opponent_valid_moves:
            # Opponent cannot move; current player wins
            self.done = True
            return self.grid.copy(), 1, True, False, {}

        # Switch to the opponent
        self.current_player *= -1
        return self.grid.copy(), 0, False, False, {}

    def get_valid_moves(self, last_move, grid):
        if last_move is None:
            # First move: All empty cells are valid
            return [i for i in range(16) if grid[i] == 0]
        else:
            # Subsequent moves: Empty cells adjacent to the last move
            row, col = last_move
            moves = []
            for r_offset, c_offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r_new, c_new = row + r_offset, col + c_offset
                if 0 <= r_new < 4 and 0 <= c_new < 4:
                    idx = r_new * 4 + c_new
                    if grid[idx] == 0:
                        moves.append(idx)
            return moves

    def valid_moves(self):
        # Return valid moves for the current player
        return self.get_valid_moves(self.last_move, self.grid)

    def render(self):
        # Generate a string representation of the grid
        grid_str = ""
        for row in range(4):
            grid_str += "+---+---+---+---+\n"
            grid_str += "|"
            for col in range(4):
                idx = row * 4 + col
                if self.grid[idx] == 1:
                    grid_str += " X |"
                elif self.grid[idx] == -1:
                    grid_str += " O |"
                else:
                    grid_str += "   |"
            grid_str += "\n"
        grid_str += "+---+---+---+---+\n"
        return grid_str
