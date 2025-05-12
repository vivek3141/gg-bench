import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.last_moves = {1: None, -1: None}  # Stores last move of each player
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}  # Game is over

        if self.board[action] != 0:
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {},
            )  # Invalid move (cell occupied)

        opponent_player = -self.current_player
        opponent_last_move = self.last_moves[opponent_player]

        # Check adjacency if opponent's last move exists
        if opponent_last_move is not None:
            if self.is_adjacent(action, opponent_last_move):
                self.done = True
                return (
                    self.board.copy(),
                    -10,
                    True,
                    False,
                    {},
                )  # Invalid move (adjacent to opponent's last move)

        # Valid move
        self.board[action] = self.current_player
        self.last_moves[self.current_player] = action

        # Check if opponent has any valid moves
        opponent_valid_moves = self.get_valid_moves(opponent_player)
        if not opponent_valid_moves:
            self.done = True
            return self.board.copy(), 1, True, False, {}  # Current player wins

        # Switch player
        self.current_player = opponent_player

        return self.board.copy(), 0, False, False, {}  # Continue game

    def render(self):
        symbols = {1: " X ", -1: " O ", 0: "   "}
        board_str = "\n-------------\n"
        for i in range(3):
            board_str += "|"
            for j in range(3):
                index = i * 3 + j
                board_str += symbols[self.board[index]] + "|"
            board_str += "\n-------------\n"
        return board_str

    def valid_moves(self):
        return self.get_valid_moves(self.current_player)

    def get_valid_moves(self, player):
        valid_actions = []
        opponent_player = -player
        last_opponent_move = self.last_moves[opponent_player]

        for action in range(9):
            if self.board[action] != 0:
                continue  # Skip occupied cells
            if last_opponent_move is not None:
                if not self.is_adjacent(action, last_opponent_move):
                    valid_actions.append(action)
            else:
                if player == 1:
                    # First move for Player 1
                    valid_actions.append(action)
                else:
                    # Player 2 cannot move before Player 1 makes a move
                    continue
        return valid_actions

    def is_adjacent(self, action1, action2):
        if action1 == action2:
            return False  # Same cell is not adjacent
        row1, col1 = divmod(action1, 3)
        row2, col2 = divmod(action2, 3)
        return abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1
