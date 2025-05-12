import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 13 squares numbered from 0 to 12
        self.action_space = spaces.Discrete(13)

        # Define observation space: array of 13 elements with values -1, 0, or 1
        # -1 for Player 2's claimed squares, 0 for unclaimed squares, 1 for Player 1's claimed squares
        self.observation_space = spaces.Box(low=-1, high=1, shape=(13,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(
            13, dtype=np.int8
        )  # All squares are unclaimed at the start
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game over flag
        return self.board.copy(), {}  # Return initial observation and info dict

    def step(self, action):
        if self.done:
            # If the game is already over, return current state with zero reward
            return self.board.copy(), 0, True, False, {}

        # Check if the current player has any valid moves
        if not self.has_valid_moves():
            # Current player cannot make a move and loses the game
            self.done = True
            return self.board.copy(), -1, True, False, {}

        # Validate the action
        if (
            action < 0
            or action >= 13
            or self.board[action] != 0
            or not self.is_valid_move(action)
        ):
            # Invalid move results in an immediate loss
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Make the move
        self.board[action] = self.current_player

        # Check if the opponent has any valid moves
        if not self.has_valid_moves():
            # Opponent cannot move; current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch turns
        self.current_player *= -1  # Toggle between 1 (Player 1) and -1 (Player 2)
        return self.board.copy(), 0, False, False, {}  # Continue the game

    def render(self):
        # Generate a string representation of the board
        board_str = ""
        for i in range(13):
            if self.board[i] == 1:
                board_str += "[P1] "
            elif self.board[i] == -1:
                board_str += "[P2] "
            else:
                board_str += f"[{i+1}] "
        return board_str.strip()

    def valid_moves(self):
        # Return a list of valid moves for the current player
        return [i for i in range(13) if self.is_valid_move(i)]

    def is_valid_move(self, action):
        # Check if the action is valid: square is unclaimed and not adjacent to any claimed square
        if self.board[action] != 0:
            return False
        for i in range(13):
            if self.board[i] != 0 and abs(i - action) == 1:
                return False
        return True

    def has_valid_moves(self):
        # Check if the current player has any valid moves
        return any(self.is_valid_move(i) for i in range(13))
