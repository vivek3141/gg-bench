import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 25 cells for movement * 25 cells for block placement = 625 possible actions
        self.action_space = spaces.Discrete(625)
        # Observation space: 5x5 grid with values:
        # 0 - empty cells
        # 1 - Player 1
        # 2 - Player 2
        # -1 - Blocked cells
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(5, 5), dtype=np.float32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board
        self.board = np.zeros((5, 5), dtype=np.float32)
        # Set initial positions
        self.player_positions = {
            1: (0, 2),  # Player 1 starts at cell 3 (index 2)
            2: (4, 2),  # Player 2 starts at cell 23 (index 22)
        }
        self.board[self.player_positions[1]] = 1
        self.board[self.player_positions[2]] = 2
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), -10, True, False, {}

        # Decode action into move and block placements
        move_cell = action // 25
        block_cell = action % 25

        move_row = move_cell // 5
        move_col = move_cell % 5
        block_row = block_cell // 5
        block_col = block_cell % 5

        # Validate move
        valid_move = self.validate_move(move_row, move_col)
        if not valid_move:
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Move the player
        self.update_player_position(move_row, move_col)

        # Validate block placement
        valid_block = self.validate_block(block_row, block_col)
        if not valid_block:
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Place the block
        self.board[block_row, block_col] = -1

        # Check for win condition
        if self.check_win():
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Check if opponent has any valid moves; if not, current player wins
        opponent = 2 if self.current_player == 1 else 1
        if not self.has_valid_moves(opponent):
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch to the next player
        self.current_player = opponent

        # Game continues
        return self.board.copy(), -10, False, False, {}

    def validate_move(self, move_row, move_col):
        # Check if move is within bounds
        if not (0 <= move_row < 5 and 0 <= move_col < 5):
            return False

        # Get current player's position
        current_row, current_col = self.player_positions[self.current_player]

        # Check if move is adjacent
        if abs(move_row - current_row) + abs(move_col - current_col) != 1:
            return False

        # Check if move cell is empty and not blocked
        if self.board[move_row, move_col] != 0:
            return False

        return True

    def update_player_position(self, move_row, move_col):
        # Remove player from current position
        current_row, current_col = self.player_positions[self.current_player]
        self.board[current_row, current_col] = 0
        # Update player's position
        self.player_positions[self.current_player] = (move_row, move_col)
        # Place player on new position
        self.board[move_row, move_col] = self.current_player

    def validate_block(self, block_row, block_col):
        # Check if block placement is within bounds
        if not (0 <= block_row < 5 and 0 <= block_col < 5):
            return False

        # Check if block cell is empty
        if self.board[block_row, block_col] != 0:
            return False

        return True

    def check_win(self):
        # Check if current player has reached the opposite side
        win_row = 4 if self.current_player == 1 else 0
        player_row, _ = self.player_positions[self.current_player]
        if player_row == win_row:
            return True
        return False

    def has_valid_moves(self, player):
        # Check if the player has any valid moves
        row, col = self.player_positions[player]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for d_row, d_col in directions:
            new_row = row + d_row
            new_col = col + d_col
            if (
                0 <= new_row < 5
                and 0 <= new_col < 5
                and self.board[new_row, new_col] == 0
            ):
                return True
        return False

    def render(self):
        # Return a string representation of the board
        board_str = ""
        for row in range(5):
            board_str += "|"
            for col in range(5):
                cell = self.board[row, col]
                if cell == 1:
                    board_str += "P1|"
                elif cell == 2:
                    board_str += "P2|"
                elif cell == -1:
                    board_str += " X|"
                else:
                    board_str += "  |"
            board_str += "\n"
        return board_str

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        # Get current player's possible moves
        row, col = self.player_positions[self.current_player]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        possible_moves = []
        for d_row, d_col in directions:
            new_row = row + d_row
            new_col = col + d_col
            if (
                0 <= new_row < 5
                and 0 <= new_col < 5
                and self.board[new_row, new_col] == 0
            ):
                move_cell = new_row * 5 + new_col
                possible_moves.append(move_cell)

        # Get possible block placements
        empty_cells = np.argwhere(self.board == 0)
        block_cells = [cell[0] * 5 + cell[1] for cell in empty_cells]

        # Combine possible moves and blocks into actions
        for move_cell in possible_moves:
            for block_cell in block_cells:
                action = move_cell * 25 + block_cell
                valid_actions.append(action)

        return valid_actions
