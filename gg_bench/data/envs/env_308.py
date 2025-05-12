import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers 1 to 20 (indices 0 to 19)
        self.action_space = spaces.Discrete(20)

        # Observation space: the number line from 1 to 20
        # Each position can be:
        # -1: Claimed by Player 2 (P2)
        #  0: Unclaimed
        #  1: Claimed by Player 1 (P1)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(20, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}  # Game is over

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return (
                self.board.copy(),
                -100,
                True,
                False,
                {},
            )  # Invalid move ends the game

        # Valid move: Claim the number
        self.board[action] = self.current_player

        # Check if opponent has any valid moves
        opponent_valid_moves = self.get_valid_moves(-self.current_player)
        if not opponent_valid_moves:
            # Opponent cannot move: current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch current player
        self.current_player *= -1

        # Reward for valid move is -10
        return self.board.copy(), -10, False, False, {}

    def render(self):
        # Render the number line
        board_str = "Number Line State:\n"
        for i in range(20):
            if self.board[i] == 1:
                board_str += " P1 "
            elif self.board[i] == -1:
                board_str += " P2 "
            else:
                board_str += f" {i+1:2} "
            if i != 19:
                board_str += "|"
        board_str += "\n"
        return board_str

    def valid_moves(self):
        # Return the list of valid moves for the current player
        return self.get_valid_moves(self.current_player)

    def get_valid_moves(self, player):
        opponent = -player
        opponent_positions = np.where(self.board == opponent)[0]
        # Get positions adjacent to opponent's claimed numbers
        adjacent_positions = set()
        for pos in opponent_positions:
            if pos - 1 >= 0:
                adjacent_positions.add(pos - 1)
            if pos + 1 < 20:
                adjacent_positions.add(pos + 1)
        # Valid moves: Unclaimed numbers not adjacent to opponent's numbers
        valid_moves = []
        for i in range(20):
            if self.board[i] == 0 and i not in adjacent_positions:
                valid_moves.append(i)
        return valid_moves
