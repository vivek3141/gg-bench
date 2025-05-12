import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.int8)

        # Initialize the environment's variables
        self.board = None  # Will be initialized in reset()
        self.current_player = None  # 1 for Player 1, -1 for Player 2
        self.last_moves = None  # Tracks the last number picked by each player
        self.terminated = False  # Tracks if the game has ended

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(12, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.last_moves = {1: None, -1: None}
        self.terminated = False
        return self.board.copy(), {}  # Return initial observation and info

    def step(self, action):
        if self.terminated:
            return self.board.copy(), 0, True, False, {}

        number = action + 1  # Convert action index to actual number (1-12)

        # Check if action is valid
        if self.board[action] != 0:
            # Invalid move: number already claimed
            self.terminated = True
            return self.board.copy(), -10, True, False, {}

        opponent = -self.current_player
        opponent_last_move = self.last_moves[opponent]

        # First move: any unclaimed number is valid
        if opponent_last_move is None:
            valid = True
        else:
            # Check adjacency
            if abs(number - opponent_last_move) != 1:
                valid = False
            else:
                # Check opposite parity
                if number % 2 == opponent_last_move % 2:
                    valid = False
                else:
                    valid = True

        if not valid:
            # Invalid move according to game rules
            self.terminated = True
            return self.board.copy(), -10, True, False, {}

        # Valid move: update the game state
        self.board[action] = self.current_player
        self.last_moves[self.current_player] = number

        # Check if the opponent has any valid moves
        opponent_valid_moves = self.get_valid_moves(-self.current_player)
        if not opponent_valid_moves:
            # Opponent cannot move: current player wins
            self.terminated = True
            return self.board.copy(), 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        return self.board.copy(), 0, False, False, {}

    def get_valid_moves(self, player):
        opponent = -player
        opponent_last_move = self.last_moves[opponent]
        valid_moves = []

        if opponent_last_move is None:
            # First move: any unclaimed number
            valid_moves = [i for i in range(12) if self.board[i] == 0]
        else:
            for i in range(12):
                if self.board[i] == 0:
                    number = i + 1
                    # Check adjacency
                    if abs(number - opponent_last_move) == 1:
                        # Check opposite parity
                        if number % 2 != opponent_last_move % 2:
                            valid_moves.append(i)
        return valid_moves

    def valid_moves(self):
        # Returns a list of valid action indices for the current player
        return self.get_valid_moves(self.current_player)

    def render(self):
        # Returns a visual representation of the game state as a string
        output = "Number Line:\n"
        for i in range(12):
            number = i + 1
            if self.board[i] == 0:
                output += f"{number:2} "
            elif self.board[i] == 1:
                output += f"P1({number:2}) "
            elif self.board[i] == -1:
                output += f"P2({number:2}) "
        return output
