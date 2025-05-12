import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # Action space is Discrete(21): indices 0-20
        # 0-10: color single position 0-10
        # 11-20: color positions (i, i+1) where i=0-9
        self.action_space = spaces.Discrete(21)
        # Observation space is the board state: array of 11 positions
        # Each position can be -1 (Blue), 0 (Uncolored), or 1 (Red)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.int8)
        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board to all uncolored positions
        self.board = np.zeros(11, dtype=np.int8)
        # Set the starting player (1 for Red, -1 for Blue)
        self.current_player = 1
        # Game is not over
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If game is already over, return current state
            return self.board.copy(), 0, True, False, {}

        # Map the action index to board positions
        if 0 <= action <= 10:
            # Single position action
            positions = [action]
        elif 11 <= action <= 20:
            # Two adjacent positions action
            pos = action - 11
            positions = [pos, pos + 1]
        else:
            # Invalid action index
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Validate the selected positions
        valid = True
        for pos in positions:
            if self.board[pos] != 0:
                valid = False
                break

        if not valid:
            # Invalid move: position(s) already colored
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Apply the action: color the positions with the current player's color
        for pos in positions:
            self.board[pos] = self.current_player

        # Check for win condition
        winner = self.check_winner()
        if winner == self.current_player:
            # Current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Check for a full board (no valid moves left)
        if np.all(self.board != 0):
            # Game over with no winner
            self.done = True
            return self.board.copy(), 0, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        return (
            self.board.copy(),
            0,
            False,
            False,
            {},
        )  # Return observation, reward, terminated, truncated, info

    def check_winner(self):
        current = self.current_player
        # Iterate over possible sequences of length three
        for i in range(0, 9):
            if (
                self.board[i] == current
                and self.board[i + 1] == current
                and self.board[i + 2] == current
            ):
                # Ensure the sequence is exactly three (not longer)
                left_ok = (i == 0) or (self.board[i - 1] != current)
                right_ok = (i + 3 == 11) or (self.board[i + 3] != current)
                if left_ok and right_ok:
                    return current  # Current player wins
        return 0  # No winner found

    def render(self):
        # Create a visual representation of the board
        chars = []
        for cell in self.board:
            if cell == 1:
                chars.append("R")
            elif cell == -1:
                chars.append("B")
            else:
                chars.append("_")
        board_str = " ".join(chars)
        return board_str  # Return the board as a string

    def valid_moves(self):
        valid_actions = []
        # Check for valid single position moves
        for i in range(11):
            if self.board[i] == 0:
                valid_actions.append(i)
        # Check for valid two adjacent positions moves
        for i in range(10):
            if self.board[i] == 0 and self.board[i + 1] == 0:
                action_index = 11 + i
                valid_actions.append(action_index)
        return valid_actions  # Return a list of valid action indices
