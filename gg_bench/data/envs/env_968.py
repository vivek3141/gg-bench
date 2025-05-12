import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 4 move directions * 16 block positions = 64 possible actions
        self.action_space = spaces.Discrete(64)
        # Observation space: 4x4 grid with values -1 (blocked), 0 (empty), 1 (P1), 2 (P2)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(4, 4), dtype=np.int8)

        # Initialize the board and game variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((4, 4), dtype=np.int8)
        # Set initial positions
        self.board[0, 0] = 1  # P1 starts at (0, 0)
        self.board[3, 3] = 2  # P2 starts at (3, 3)
        self.positions = {
            1: (0, 0),  # P1 position
            2: (3, 3),  # P2 position
        }
        self.current_player = 1  # P1 starts
        self.done = False
        return self.board.copy(), {}  # Return a copy of the board and empty info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}
        # Decode action
        move_direction = action // 16
        block_index = action % 16
        move_offsets = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }  # Up, Down, Left, Right

        # Get current player position
        row, col = self.positions[self.current_player]

        # Calculate new position
        d_row, d_col = move_offsets.get(move_direction, (0, 0))
        new_row, new_col = row + d_row, col + d_col

        # Validate move
        if not (0 <= new_row < 4 and 0 <= new_col < 4):
            # Move is off the grid
            reward = -10
            self.done = True
            return self.board.copy(), reward, True, False, {}
        if self.board[new_row, new_col] != 0:
            # Cell is not empty
            reward = -10
            self.done = True
            return self.board.copy(), reward, True, False, {}

        # Validate block
        block_row, block_col = divmod(block_index, 4)
        if not (0 <= block_row < 4 and 0 <= block_col < 4):
            # Block position is off the grid
            reward = -10
            self.done = True
            return self.board.copy(), reward, True, False, {}
        if self.board[block_row, block_col] != 0:
            # Cannot block occupied or already blocked cell
            reward = -10
            self.done = True
            return self.board.copy(), reward, True, False, {}

        # Perform move
        self.board[row, col] = 0
        self.board[new_row, new_col] = self.current_player
        self.positions[self.current_player] = (new_row, new_col)

        # Perform block
        self.board[block_row, block_col] = -1  # Mark as blocked

        # Check if opponent has any valid moves
        opponent = 2 if self.current_player == 1 else 1
        opponent_row, opponent_col = self.positions[opponent]
        opponent_moves = self._get_valid_directions(opponent_row, opponent_col)

        if not opponent_moves:
            # Opponent has no moves; current player wins
            reward = 1
            self.done = True
            return self.board.copy(), reward, True, False, {}

        # Switch player
        self.current_player = opponent
        reward = 0
        return self.board.copy(), reward, False, False, {}

    def render(self):
        board_str = ""
        symbols = {0: "·", -1: "X", 1: "P1", 2: "P2"}
        for i in range(4):
            for j in range(4):
                cell = self.board[i, j]
                board_str += f"{symbols[cell]:>2} "
            board_str += "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        # Get current player position
        row, col = self.positions[self.current_player]
        valid_directions = self._get_valid_directions(row, col)
        if not valid_directions:
            return valid_actions  # No valid moves

        # Get list of empty cells for blocking
        empty_cells = np.argwhere(self.board == 0)

        for move_dir in valid_directions:
            for cell in empty_cells:
                block_index = cell[0] * 4 + cell[1]
                action = move_dir * 16 + block_index
                valid_actions.append(action)
        return valid_actions

    def _get_valid_directions(self, row, col):
        move_offsets = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }  # Up, Down, Left, Right
        valid_directions = []
        for dir_idx, (d_row, d_col) in move_offsets.items():
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < 4 and 0 <= new_col < 4:
                if self.board[new_row, new_col] == 0:
                    valid_directions.append(dir_idx)
        return valid_directions
