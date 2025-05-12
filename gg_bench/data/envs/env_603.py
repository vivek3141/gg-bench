import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space consists of all possible squares (0 to 24)
        self.action_space = spaces.Discrete(25)

        # The observation space is the board state
        # -1: Blocked square, 0: Empty, 1: Player 1's Knight, 2: Player 2's Knight
        self.observation_space = spaces.Box(low=-1, high=2, shape=(25,), dtype=np.int8)

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the board
        # 5x5 grid flattened to 1D array
        self.board = np.zeros(25, dtype=np.int8)

        # Set player positions
        self.player_positions = {1: 0, 2: 24}

        # Place Player 1's Knight at A1 (index 0)
        self.board[0] = 1

        # Place Player 2's Knight at E5 (index 24)
        self.board[24] = 2

        # Set the current player
        self.current_player = 1

        # Game not done
        self.done = False

        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), -10, True, False, {}

        current_pos = self.player_positions[self.current_player]

        valid_moves = self.get_valid_moves(current_pos)

        if action not in valid_moves:
            # Invalid move
            self.done = True
            return self.board.copy(), -10, True, False, {}

        opponent = 1 if self.current_player == 2 else 2

        # Move the knight
        self.board[current_pos] = -1  # Block the square we moved from
        self.board[action] = self.current_player
        self.player_positions[self.current_player] = action

        # Check if capturing the opponent's knight
        if action == self.player_positions[opponent]:
            self.board[self.player_positions[opponent]] = 0  # Remove opponent's knight
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Check if opponent has any valid moves left
        opponent_valid_moves = self.get_valid_moves(self.player_positions[opponent])
        if not opponent_valid_moves:
            # Opponent has no legal moves
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch to the next player
        self.current_player = opponent

        return self.board.copy(), 0, False, False, {}

    def render(self):
        board_str = "  A B C D E\n"
        for row in range(5):
            board_str += str(row + 1) + " "
            for col in range(5):
                pos = row * 5 + col
                val = self.board[pos]
                if val == -1:
                    board_str += "X "
                elif val == 0:
                    board_str += ". "
                elif val == 1:
                    board_str += "1 "
                elif val == 2:
                    board_str += "2 "
            board_str += "\n"
        return board_str

    def valid_moves(self):
        current_pos = self.player_positions[self.current_player]
        return self.get_valid_moves(current_pos)

    def get_valid_moves(self, position):
        valid_moves = []
        row = position // 5  # 0 to 4
        col = position % 5  # 0 to 4

        # Define the 8 possible 'L' shape moves
        moves = [
            (-2, -1),
            (-2, 1),  # Up 2, Left/Right 1
            (2, -1),
            (2, 1),  # Down 2, Left/Right 1
            (-1, -2),
            (1, -2),  # Left 2, Up/Down 1
            (-1, 2),
            (1, 2),  # Right 2, Up/Down 1
        ]

        for dr, dc in moves:
            new_row = row + dr
            new_col = col + dc
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                new_pos = new_row * 5 + new_col
                if self.board[new_pos] == -1:
                    continue  # Blocked square
                valid_moves.append(new_pos)

        return valid_moves
