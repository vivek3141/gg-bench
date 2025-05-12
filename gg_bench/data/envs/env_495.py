import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The grid is 4x4, so we have 16 possible actions (cells)
        self.action_space = spaces.Discrete(16)
        # Observation space is a 4x4 grid with values -1 (O), 0 (empty), 1 (X)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4, 4), dtype=np.int8)

        # Initialize the board and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((4, 4), dtype=np.int8)
        # Player 1 is 1 (X), Player 2 is -1 (O)
        self.current_player = 1  # Player 1 starts
        self.last_moves = {
            1: None,
            -1: None,
        }  # Keep track of last moves for each player
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Get list of valid moves for the current player
        valid_moves = self.valid_moves()

        # Check if the current player has any valid moves
        if not valid_moves:
            # Current player cannot move, they lose
            reward = -1
            self.done = True
            return self.board.copy(), reward, True, False, {}

        # Check if the action is valid
        if action not in valid_moves:
            # Invalid move
            reward = -10
            return self.board.copy(), reward, False, False, {}

        # Make the move
        row = action // 4
        col = action % 4
        self.board[row, col] = self.current_player
        self.last_moves[self.current_player] = (row, col)

        # Store the player who just moved
        previous_player = self.current_player

        # Switch to the next player
        self.current_player *= -1

        # Check if the next player has any valid moves
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            # Next player cannot move, previous player wins
            reward = 1
            self.done = True
            # No need to switch back the current player as the game is over
            return self.board.copy(), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            return self.board.copy(), reward, False, False, {}

    def render(self):
        # Create a string representation of the board
        symbols = {1: "X", -1: "O", 0: " "}
        board_str = ""
        for i in range(4):
            row_str = "|"
            for j in range(4):
                cell_value = self.board[i, j]
                row_str += f" {symbols[cell_value]} |"
            board_str += "-" * 17 + "\n" + row_str + "\n"
        board_str += "-" * 17
        return board_str

    def valid_moves(self):
        # Get the last move made by the opponent
        opponent_player = -self.current_player
        opponent_last_move = self.last_moves.get(opponent_player)

        valid_moves = []

        if opponent_last_move is None:
            if self.current_player == 1:
                # Player 1's first move: any empty cell
                for action in range(16):
                    row = action // 4
                    col = action % 4
                    if self.board[row, col] == 0:
                        valid_moves.append(action)
            else:
                # Player 2's turn but Player 1 hasn't moved (should not happen)
                pass
        else:
            # Must place adjacent to the opponent's last move
            row, col = opponent_last_move
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < 4 and 0 <= c < 4:
                    if self.board[r, c] == 0:
                        action = r * 4 + c
                        valid_moves.append(action)

        return valid_moves
