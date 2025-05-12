import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 25 possible actions (places to put the token)
        self.action_space = spaces.Discrete(25)
        # Observation is the board state: 5x5 grid
        # Each cell can be: -1 (Player 2), 0 (empty), 1 (Player 1)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.last_move_p1 = None
        self.last_move_p2 = None
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        row = action // 5
        col = action % 5

        # Check if action is valid
        if not (0 <= row < 5 and 0 <= col < 5):
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {"InvalidAction": "Out of bounds"},
            )

        if self.board[row][col] != 0:
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {"InvalidAction": "Cell already occupied"},
            )

        # Compute forbidden cells for current player
        forbidden_cells = self.get_forbidden_cells(self.current_player)

        if forbidden_cells[row][col]:
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {"InvalidAction": "Cell is forbidden"},
            )

        # Place the token
        self.board[row][col] = self.current_player

        # Update last move for current player
        if self.current_player == 1:
            self.last_move_p1 = (row, col)
        else:
            self.last_move_p2 = (row, col)

        # Check if opponent has any valid moves
        opponent = -self.current_player
        opponent_valid_moves = self.get_valid_moves(opponent)

        if not opponent_valid_moves:
            # Opponent has no valid moves, current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch to opponent
        self.current_player = opponent

        return self.board.copy(), 0, False, False, {}

    def render(self):
        board_str = ""
        for row in self.board:
            board_str += "|"
            for cell in row:
                if cell == 1:
                    board_str += " X |"
                elif cell == -1:
                    board_str += " O |"
                else:
                    board_str += "   |"
            board_str += "\n" + "-" * (5 * 4 + 1) + "\n"
        return board_str

    def valid_moves(self):
        return self.get_valid_action_indices(self.current_player)

    def get_forbidden_cells(self, player):
        opponent_last_move = self.last_move_p2 if player == 1 else self.last_move_p1
        forbidden_cells = np.zeros((5, 5), dtype=bool)

        if opponent_last_move is not None:
            row, col = opponent_last_move
            # Get all adjacent positions
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r = row + dr
                    c = col + dc
                    if 0 <= r < 5 and 0 <= c < 5:
                        forbidden_cells[r][c] = True
        return forbidden_cells

    def get_valid_moves(self, player):
        forbidden_cells = self.get_forbidden_cells(player)
        valid_moves = []
        for row in range(5):
            for col in range(5):
                if self.board[row][col] == 0 and not forbidden_cells[row][col]:
                    valid_moves.append((row, col))
        return valid_moves

    def get_valid_action_indices(self, player):
        forbidden_cells = self.get_forbidden_cells(player)
        valid_actions = []
        for idx in range(25):
            row = idx // 5
            col = idx % 5
            if self.board[row][col] == 0 and not forbidden_cells[row][col]:
                valid_actions.append(idx)
        return valid_actions
