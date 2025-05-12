import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is 16 discrete actions (positions on the 4x4 grid)
        self.action_space = spaces.Discrete(16)
        # The observation space is a 4x4 grid with values -1 (Player 2), 0 (empty), 1 (Player 1)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4, 4), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board as a 4x4 grid of zeros (empty cells)
        self.board = np.zeros((4, 4), dtype=np.int8)
        # Set the current player (1 for Player 1, -1 for Player 2)
        self.current_player = 1
        # The game is not over
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}  # Game is over

        if action < 0 or action >= 16:
            # Invalid action (out of bounds)
            self.done = True
            return self.board.copy(), -10, True, False, {}  # Invalid move

        row = action // 4
        col = action % 4

        if self.board[row, col] != 0:
            # Cell is already occupied
            self.done = True
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Check adjacency to current player's markers
        current_positions = np.argwhere(self.board == self.current_player)
        adjacent = False
        for pos in current_positions:
            if max(abs(row - pos[0]), abs(col - pos[1])) <= 1:
                adjacent = True
                break

        if adjacent:
            # Cannot place adjacent to own markers
            self.done = True
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Place the marker
        self.board[row, col] = self.current_player

        # Check if the opponent has any valid moves
        opponent = -self.current_player
        opponent_valid_moves = self.valid_moves(player=opponent)
        if not opponent_valid_moves:
            # Opponent has no valid moves, current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}  # Current player wins

        # Switch current player
        self.current_player *= -1
        return self.board.copy(), 0, False, False, {}  # Game continues

    def render(self):
        board_str = "   1 2 3 4\n  ---------\n"
        for i in range(4):
            board_str += f"{i+1} |"
            for j in range(4):
                if self.board[i, j] == 1:
                    board_str += " X"
                elif self.board[i, j] == -1:
                    board_str += " O"
                else:
                    board_str += " ."
            board_str += "\n"
        return board_str

    def valid_moves(self, player=None):
        if player is None:
            player = self.current_player
        valid_actions = []
        empty_positions = np.argwhere(self.board == 0)
        player_positions = np.argwhere(self.board == player)
        for pos in empty_positions:
            row, col = pos
            adjacent = False
            for p_pos in player_positions:
                p_row, p_col = p_pos
                if max(abs(row - p_row), abs(col - p_col)) <= 1:
                    adjacent = True
                    break
            if not adjacent:
                action = row * 4 + col
                valid_actions.append(action)
        return valid_actions
