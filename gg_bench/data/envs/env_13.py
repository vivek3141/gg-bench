import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 for 'move', 1 for 'swap', 2 for 'pass'
        self.action_space = spaces.Discrete(3)

        # Define observation space: 7 positions on the board
        # Each position can be:
        # 0: Empty
        # 1: Player 1's Marker
        # 2: Player 2's Marker
        # 3: King Piece
        self.observation_space = spaces.Box(low=0, high=3, shape=(7,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board
        self.board = np.zeros(7, dtype=np.int8)
        # Place Player Markers and King
        self.board[0] = 1  # Player 1 Marker at Position 1 (index 0)
        self.board[3] = 3  # King at Position 4 (index 3)
        self.board[6] = 2  # Player 2 Marker at Position 7 (index 6)

        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return self.board.copy(), -10, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Get current positions
        player_pos = np.where(self.board == self.current_player)[0][0]
        king_pos = np.where(self.board == 3)[0][0]

        if action == 0:
            # Move Marker one position closer to the King
            direction = 1 if self.current_player == 1 else -1
            new_pos = player_pos + direction
            self.board[player_pos] = 0
            self.board[new_pos] = self.current_player

        elif action == 1:
            # Swap positions with the King
            self.board[player_pos], self.board[king_pos] = (
                self.board[king_pos],
                self.board[player_pos],
            )

        elif action == 2:
            # Pass turn
            pass

        # Check for win condition
        king_pos = np.where(self.board == 3)[0][0]
        if self.current_player == 1 and king_pos == 6:
            # Player 1 wins
            self.done = True
            return self.board.copy(), 1, True, False, {}
        elif self.current_player == 2 and king_pos == 0:
            # Player 2 wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1
        return self.board.copy(), 0, False, False, {}

    def render(self):
        # Generate a string representation of the board
        display = []
        for pos in self.board:
            if pos == 0:
                display.append("[   ]")
            elif pos == 1:
                display.append("[ P1 ]")
            elif pos == 2:
                display.append("[ P2 ]")
            elif pos == 3:
                display.append("[ K ]")
        board_str = " ".join(display)
        board_str += "\nPositions: 1 2 3 4 5 6 7"
        return board_str

    def valid_moves(self):
        if self.done:
            return []

        valid_actions = []
        player_pos = np.where(self.board == self.current_player)[0][0]
        king_pos = np.where(self.board == 3)[0][0]
        opponent_marker = 2 if self.current_player == 1 else 1

        # Move towards the King
        direction = 1 if self.current_player == 1 else -1
        new_pos = player_pos + direction
        if (
            0 <= new_pos <= 6
            and self.board[new_pos] != opponent_marker
            and self.board[new_pos] != 3
        ):
            valid_actions.append(0)  # 'move'

        # Swap with King if adjacent
        if abs(player_pos - king_pos) == 1:
            valid_actions.append(1)  # 'swap'

        if not valid_actions:
            # No valid moves; 'pass' is allowed
            valid_actions.append(2)  # 'pass'

        return valid_actions
