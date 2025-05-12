import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(25) for the 5x5 grid positions
        self.action_space = spaces.Discrete(25)

        # The observation is a 5x5 grid with values:
        # 0 for empty, 1 for Player 1's marker, -1 for Player 2's marker
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # 1 for Player 1 ('X'), -1 for Player 2 ('O')
        self.last_move = None  # Last move made by the opponent
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Map action to board position
        row = action // 5  # actions 0-24 represent positions on the board
        col = action % 5

        # Check if the action is valid
        if not self.is_valid_action(action):
            self.done = True
            return self.board.copy(), -10, True, False, {}  # Invalid move, player loses

        # Place the marker
        self.board[row, col] = self.current_player

        # Check for victory
        if row == 2 and col == 2:  # Captured the central block (3,3)
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Set last move to current action
        self.last_move = (row, col)

        # Check if opponent has any valid moves
        opponent = -self.current_player
        opponent_valid_moves = self.get_valid_moves_for_player(opponent)
        if len(opponent_valid_moves) == 0:
            # Opponent cannot move, current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch players
        self.current_player = opponent

        return self.board.copy(), 0, False, False, {}

    def is_valid_action(self, action):
        return self.is_valid_action_for_player(
            action, self.current_player, self.last_move
        )

    def is_valid_action_for_player(self, action, player, opponent_last_move):
        row = action // 5
        col = action % 5

        # Check if position is empty
        if self.board[row, col] != 0:
            return False

        if opponent_last_move is None:
            # First move
            if player == 1:
                # Must be on the edge
                if row == 0 or row == 4 or col == 0 or col == 4:
                    return True
                else:
                    return False
            else:
                # Player 2 cannot move before Player 1
                return False
        else:
            # Must be adjacent to opponent's last move
            last_row, last_col = opponent_last_move
            if (abs(row - last_row) == 1 and col == last_col) or (
                abs(col - last_col) == 1 and row == last_row
            ):
                # Adjacent horizontally or vertically
                return True

        return False

    def get_valid_moves_for_player(self, player):
        valid_moves = []
        opponent_last_move = self.last_move

        for action in range(25):
            if self.is_valid_action_for_player(action, player, opponent_last_move):
                valid_moves.append(action)

        return valid_moves

    def valid_moves(self):
        return self.get_valid_moves_for_player(self.current_player)

    def render(self):
        # Return a string representation of the board
        symbols = {0: " ", 1: "X", -1: "O"}
        board_str = "    1   2   3   4   5\n  +---+---+---+---+---+\n"
        for i in range(5):
            row_str = f"{i+1} |"
            for j in range(5):
                row_str += f" {symbols[self.board[i, j]]} |"
            row_str += "\n  +---+---+---+---+---+\n"
            board_str += row_str
        return board_str
