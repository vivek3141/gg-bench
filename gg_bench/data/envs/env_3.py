import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(25)  # 5x5 grid has 25 possible actions
        self.observation_space = spaces.Box(low=-1, high=1, shape=(25,), dtype=np.int8)

        # Initialize the board and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(25, dtype=np.int8)  # Flattened 5x5 grid
        self.current_player = 1  # Player 1 is 1, Player 2 is -1
        self.last_move = None  # No last move at the start
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Validate the action index
        if action < 0 or action >= 25:
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Check if the selected cell is unclaimed
        if self.board[action] != 0:
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Check move validity according to the game rules
        if self.last_move is not None:
            # Get coordinates of the opponent's last move
            opponent_row = self.last_move // 5
            opponent_col = self.last_move % 5
            # Get coordinates of the current action
            action_row = action // 5
            action_col = action % 5
            # The move must be in the same row or column as the opponent's last move
            if action_row != opponent_row and action_col != opponent_col:
                self.done = True
                return self.board.copy(), -10, True, False, {}

        # Apply the action
        self.board[action] = self.current_player
        self.last_move = action  # Update the last move

        # Check if the opponent has any valid moves
        opponent_has_moves = False
        opponent_valid_moves = []
        unclaimed_cells = np.where(self.board == 0)[0]
        # Coordinates of the player's last move (which will be the opponent's reference)
        player_row = self.last_move // 5
        player_col = self.last_move % 5

        for cell in unclaimed_cells:
            cell_row = cell // 5
            cell_col = cell % 5
            if cell_row == player_row or cell_col == player_col:
                opponent_has_moves = True
                opponent_valid_moves.append(cell)
                break  # No need to find all moves, one is enough

        if not opponent_has_moves:
            # Opponent has no valid moves, current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        return self.board.copy(), 0, False, False, {}

    def render(self):
        # Generate a string representation of the board
        board_str = "  1 2 3 4 5\n"
        for i in range(5):
            row_str = chr(ord("A") + i) + " "
            for j in range(5):
                idx = i * 5 + j
                if self.board[idx] == 1:
                    row_str += "X "
                elif self.board[idx] == -1:
                    row_str += "O "
                else:
                    row_str += ". "
            board_str += row_str.strip() + "\n"
        return board_str

    def valid_moves(self):
        # Return a list of valid moves for the current player
        valid_moves = []
        unclaimed_cells = np.where(self.board == 0)[0]

        if self.last_move is None:
            # First move: any cell is valid
            valid_moves = unclaimed_cells.tolist()
        else:
            # Subsequent moves: cells in the same row or column as opponent's last move
            opponent_row = self.last_move // 5
            opponent_col = self.last_move % 5
            for cell in unclaimed_cells:
                cell_row = cell // 5
                cell_col = cell % 5
                if cell_row == opponent_row or cell_col == opponent_col:
                    valid_moves.append(cell)

        return valid_moves
