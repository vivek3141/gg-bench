import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    This is an implementation of the 'Blockade' game.
    The board is a 5x5 grid.
    """

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(25), corresponding to the 25 cells on the board
        self.action_space = spaces.Discrete(25)
        # The observation is a 25-dimensional vector representing the board state
        # 1 represents Player 1's token, -1 represents Player 2's token, 0 represents empty cell
        self.observation_space = spaces.Box(low=-1, high=1, shape=(25,), dtype=np.int8)

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(25, dtype=np.int8)  # 5x5 board flattened
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        # If the game is already over
        if self.done:
            return self.board.copy(), 0, True, False, {}

        valid_moves = self.valid_moves(self.current_player)

        # Check if action is valid
        if action not in valid_moves:
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {"invalid_move": True},
            )  # Observation, reward, terminated, truncated, info

        # Place the current player's token
        self.board[action] = self.current_player

        # Check if opponent has any valid moves
        opponent = -self.current_player
        opponent_valid_moves = self.valid_moves(opponent)

        if not opponent_valid_moves:
            # Opponent has no valid moves, current player wins
            self.done = True
            reward = 1  # Current player wins
            return self.board.copy(), reward, True, False, {}
        else:
            # Switch to opponent's turn
            self.current_player = opponent
            return self.board.copy(), 0, False, False, {}

    def render(self):
        board_str = ""
        for row in range(5):
            row_str = ""
            for col in range(5):
                index = row * 5 + col
                cell = self.board[index]
                if cell == 1:
                    cell_str = "X"
                elif cell == -1:
                    cell_str = "O"
                else:
                    cell_str = " "
                row_str += f"({row+1},{col+1})[{cell_str}] "
            board_str += row_str.strip() + "\n"
        return board_str.strip()

    def valid_moves(self, player):
        valid_moves = []
        for index in range(25):
            if self.board[index] != 0:
                continue  # Cell is not empty
            # Check adjacency to opponent's tokens
            if not self.is_adjacent_to_opponent(index, player):
                valid_moves.append(index)
        return valid_moves

    def is_adjacent_to_opponent(self, index, player):
        opponent = -player
        row = index // 5
        col = index % 5
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip the current cell
                r = row + dr
                c = col + dc
                if 0 <= r < 5 and 0 <= c < 5:
                    neighbor_index = r * 5 + c
                    if self.board[neighbor_index] == opponent:
                        return True  # Adjacent to opponent
        return False
