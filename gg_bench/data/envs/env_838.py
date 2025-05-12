import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: Each action corresponds to moving the knight to one of the 64 squares
        # Squares are numbered from 0 to 63
        self.action_space = spaces.Discrete(64)

        # Define observation space
        # Each square can be:
        # 0 - Active square (empty and available for movement)
        # 1 - Removed square (previously occupied and no longer available)
        # 2 - Player 1's knight
        # 3 - Player 2's knight
        self.observation_space = spaces.Box(low=0, high=3, shape=(8, 8), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the board with all squares active
        self.board = np.zeros((8, 8), dtype=np.int8)

        # Set the initial positions of the knights
        self.player_positions = {
            1: (0, 0),  # Player 1 (White) starts at (1, 1) which is index (0, 0)
            2: (7, 7),  # Player 2 (Black) starts at (8, 8) which is index (7, 7)
        }

        # Mark the knights on the board
        self.board[0, 0] = 2  # Player 1's knight
        self.board[7, 7] = 3  # Player 2's knight

        # Set current player (Player 1 starts)
        self.current_player = 1

        self.done = False

        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), -10, True, False, {}  # Game is already over

        # Check if current player has any valid moves
        if not self.valid_moves():
            # Current player cannot move and loses the game
            self.done = True
            return self.board.copy(), 0, True, False, {}  # No reward for losing

        # Convert action to (row, col)
        row = action // 8
        col = action % 8
        dest = (row, col)

        # Get current player's knight position
        curr_pos = self.player_positions[self.current_player]

        # Validate the move
        if not self.is_valid_knight_move(curr_pos, dest):
            # Invalid move results in a large negative reward and ends the game
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Move is valid, update the board
        # Remove square where the knight currently is
        self.board[curr_pos] = 1  # Mark as removed
        # Place knight in new position
        self.board[dest] = 2 if self.current_player == 1 else 3
        # Update player's knight position
        self.player_positions[self.current_player] = dest

        # Check if opponent has any valid moves
        opponent = 2 if self.current_player == 1 else 1
        opponent_pos = self.player_positions[opponent]
        opponent_moves = self.get_valid_moves(opponent_pos)

        if not opponent_moves:
            # Opponent cannot move; current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}  # Reward for winning

        # Switch to the opponent
        self.current_player = opponent

        return self.board.copy(), 0, False, False, {}  # Continue the game

    def render(self):
        # Return a string representation of the board
        symbols = {0: ".", 1: "X", 2: "N1", 3: "N2"}
        board_str = ""
        for i in range(7, -1, -1):  # Rows from top (7) to bottom (0)
            row_str = ""
            for j in range(8):
                val = self.board[i, j]
                s = symbols[val]
                row_str += f"{s:>3}"
            board_str += row_str + "\n"
        return board_str

    def valid_moves(self):
        # Return a list of valid moves as action indices for the current player
        curr_pos = self.player_positions[self.current_player]
        moves = self.get_valid_moves(curr_pos)
        actions = [r * 8 + c for (r, c) in moves]
        return actions

    def is_valid_knight_move(self, start, end):
        # Check if the move is a valid knight's move and the destination is active
        row_diff = abs(end[0] - start[0])
        col_diff = abs(end[1] - start[1])

        # Check for L-shaped movement
        if (row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2):
            # Check if destination is within bounds
            if 0 <= end[0] < 8 and 0 <= end[1] < 8:
                # Destination square must be active (not removed or occupied)
                if self.board[end[0], end[1]] == 0:
                    return True
        return False

    def get_valid_moves(self, position):
        # Calculate all valid moves from the current position
        moves = []
        directions = [
            (2, 1),
            (1, 2),
            (-1, 2),
            (-2, 1),
            (-2, -1),
            (-1, -2),
            (1, -2),
            (2, -1),
        ]
        for d in directions:
            new_r = position[0] + d[0]
            new_c = position[1] + d[1]
            if 0 <= new_r < 8 and 0 <= new_c < 8:
                # Check if the destination square is active
                if self.board[new_r, new_c] == 0:
                    moves.append((new_r, new_c))
        return moves
