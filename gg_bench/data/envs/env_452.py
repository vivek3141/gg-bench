import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The board is 5x5, so there are 25 possible actions (0 to 24)
        self.action_space = spaces.Discrete(25)
        # Observation space represents the state of the board
        # -1 for Player 2's symbol ('O'), 0 for empty, 1 for Player 1's symbol ('X')
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        # Initialize the board and game variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.last_moves = {1: None, -1: None}  # Store the last move of each player
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return self.board.copy(), 0, True, False, {}

        valid_moves = self.valid_moves()
        if not valid_moves:
            # Current player has no valid moves and loses
            self.done = True
            reward = -10
            terminated = True
            truncated = False
            return self.board.copy(), reward, terminated, truncated, {}

        if action not in valid_moves:
            # Invalid move
            self.done = True
            reward = -10
            terminated = True
            truncated = False
            return self.board.copy(), reward, terminated, truncated, {}

        # Apply the move
        row, col = divmod(action, 5)
        self.board[row, col] = self.current_player
        self.last_moves[self.current_player] = (row, col)

        # Check if the opponent has valid moves
        opponent = -self.current_player
        opponent_valid_moves = self.valid_moves_for_player(opponent)

        if not opponent_valid_moves:
            # Opponent has no valid moves; current player wins
            self.done = True
            reward = 1
            terminated = True
            truncated = False
            return self.board.copy(), reward, terminated, truncated, {}
        else:
            # Switch to opponent
            self.current_player = opponent
            reward = 0
            terminated = False
            truncated = False
            return self.board.copy(), reward, terminated, truncated, {}

    def render(self):
        board_str = ""
        symbol_map = {1: "X", -1: "O", 0: "."}
        for row in range(5):
            for col in range(5):
                board_str += symbol_map[self.board[row, col]] + " "
            board_str += "\n"
        return board_str

    def valid_moves(self):
        return self.valid_moves_for_player(self.current_player)

    def valid_moves_for_player(self, player):
        valid_actions = []
        opponent = -player
        last_opponent_move = self.last_moves[opponent]

        if last_opponent_move is None:
            # First move of the game
            if player == 1:
                # Player 1 can move anywhere
                for idx in range(25):
                    row, col = divmod(idx, 5)
                    if self.board[row, col] == 0:
                        valid_actions.append(idx)
            else:
                # Should not happen in normal play
                pass
        else:
            row, col = last_opponent_move
            # Check adjacent cells
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < 5 and 0 <= c < 5:
                    if self.board[r, c] == 0:
                        idx = r * 5 + c
                        valid_actions.append(idx)
        return valid_actions
