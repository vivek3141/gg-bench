import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(10,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(10, dtype=np.float32)
        self.current_player = 1  # 1 for 'X', -1 for 'O'
        self.done = False
        return self.board, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board, 0, True, False, {}

        # Check if action is valid (within the board)
        if action < 0 or action >= 10:
            return self.board, -10, True, False, {}

        # Check if the cell is empty
        if self.board[action] != 0:
            return self.board, -10, True, False, {}

        # Check adjacency restriction
        opponent = -self.current_player
        adjacent_indices = []
        if action > 0:
            adjacent_indices.append(action - 1)
        if action < 9:
            adjacent_indices.append(action + 1)

        illegal_move = False
        for idx in adjacent_indices:
            if self.board[idx] == opponent:
                illegal_move = True
                break

        if illegal_move:
            return self.board, -10, True, False, {}

        # Place the token
        self.board[action] = self.current_player

        # Check if the next player has any valid moves
        self.current_player *= -1  # Switch player
        if not self.has_valid_moves():
            self.done = True
            reward = 1  # Current player wins
            return self.board, reward, True, False, {}

        return self.board, 0, False, False, {}

    def has_valid_moves(self):
        """Check if the current player has any valid moves."""
        opponent = -self.current_player
        for action in range(10):
            if self.board[action] != 0:
                continue

            # Check adjacency restriction
            adjacent_indices = []
            if action > 0:
                adjacent_indices.append(action - 1)
            if action < 9:
                adjacent_indices.append(action + 1)

            illegal_move = False
            for idx in adjacent_indices:
                if self.board[idx] == opponent:
                    illegal_move = True
                    break

            if not illegal_move:
                return True  # At least one valid move available
        return False  # No valid moves available

    def render(self):
        """Return a visual representation of the game state as a string."""
        grid = "Positions: " + " ".join([str(i + 1) for i in range(10)]) + "\n"
        cells = "Cells:     "
        for i in range(10):
            if self.board[i] == 1:
                cells += "X "
            elif self.board[i] == -1:
                cells += "O "
            else:
                cells += "_ "
        grid += cells
        return grid

    def valid_moves(self):
        """Return a list of valid action indices for the current player."""
        valid_actions = []
        opponent = -self.current_player
        for action in range(10):
            if self.board[action] != 0:
                continue

            # Check adjacency restriction
            adjacent_indices = []
            if action > 0:
                adjacent_indices.append(action - 1)
            if action < 9:
                adjacent_indices.append(action + 1)

            illegal_move = False
            for idx in adjacent_indices:
                if self.board[idx] == opponent:
                    illegal_move = True
                    break

            if not illegal_move:
                valid_actions.append(action)
        return valid_actions
