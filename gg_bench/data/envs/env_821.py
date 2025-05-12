import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.board_size = 11  # Positions from 1 to 11
        self.board = None
        self.current_player = None
        self.done = None
        self.first_move = None  # Track if it's the first move for each player
        self.player_positions = None  # Track the positions of each player

        # Action space: positions 0 to 10 (representing positions 1 to 11)
        self.action_space = spaces.Discrete(self.board_size)

        # Observation space: array of 11 positions with values -1, 0, 1
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.board_size,), dtype=np.int8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(self.board_size, dtype=np.int8)  # Positions from 0 to 10
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.first_move = {1: True, -1: True}
        self.player_positions = {1: None, -1: None}
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over
            return self.board.copy(), 0, True, False, {}

        reward = 0
        terminated = False

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self.board.copy(), reward, True, False, {}

        # Perform the action
        if self.first_move[self.current_player]:
            # First move: place the marker
            self.board[action] = self.current_player
            self.player_positions[self.current_player] = action
            self.first_move[self.current_player] = False
        else:
            # Subsequent move: move to an adjacent position
            current_pos = self.player_positions[self.current_player]
            # Remove marker from current position
            self.board[current_pos] = 0
            # Place marker on new position
            self.board[action] = self.current_player
            self.player_positions[self.current_player] = action

        # Check if opponent can make any valid moves
        opponent = -self.current_player
        opponent_valid_moves = self.valid_moves(player=opponent)
        if len(opponent_valid_moves) == 0:
            # Current player wins
            self.done = True
            reward = 1
            return self.board.copy(), reward, True, False, {}

        # Switch to opponent's turn
        self.current_player = opponent

        return self.board.copy(), reward, False, False, {}

    def render(self):
        # Build the number line as a string
        line = ""
        for idx in range(self.board_size):
            position = self.board[idx]
            if position == 1:
                marker = "X"
            elif position == -1:
                marker = "O"
            else:
                marker = "_"
            line += marker + " "
        return line.strip()  # Return the number line string

    def valid_moves(self, player=None):
        if player is None:
            player = self.current_player

        valid_actions = []

        if self.first_move[player]:
            # First move: any unoccupied position
            valid_actions = [i for i in range(self.board_size) if self.board[i] == 0]
        else:
            # Subsequent moves: adjacent unoccupied positions
            current_pos = self.player_positions[player]
            potential_moves = [current_pos - 1, current_pos + 1]
            for pos in potential_moves:
                if 0 <= pos < self.board_size and self.board[pos] == 0:
                    valid_actions.append(pos)

        return valid_actions
