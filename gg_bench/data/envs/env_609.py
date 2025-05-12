import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(16,), dtype=np.int8)

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(16, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.player_positions = {1: None, 2: None}
        self.initial_placement_phase = True
        self.done = False
        return self.board.copy(), {}

    def step(self, action):
        if self.done:
            return self.board.copy(), -10, True, False, {}

        # Check if initial placement phase
        if self.initial_placement_phase:
            if self.board[action] != 0:
                # Invalid move: Cell already occupied
                self.done = True
                return self.board.copy(), -10, True, False, {}
            else:
                # Place the marker
                self.board[action] = self.current_player
                self.player_positions[self.current_player] = action
                if self.current_player == 1:
                    # Switch to Player 2's initial placement
                    self.current_player = 2
                else:
                    # End of initial placement phase
                    self.initial_placement_phase = False
                    self.current_player = 1
                return self.board.copy(), -10, False, False, {}
        else:
            # Check if current player has any valid moves
            valid_moves = self.valid_moves()
            if not valid_moves:
                # Current player has no valid moves: Lose
                self.done = True
                return self.board.copy(), -1, True, False, {}

            if action not in valid_moves:
                # Invalid move
                self.done = True
                return self.board.copy(), -10, True, False, {}

            # Move the marker
            prev_pos = self.player_positions[self.current_player]
            self.board[prev_pos] = -1  # Mark the previous cell as occupied
            self.board[action] = self.current_player
            self.player_positions[self.current_player] = action

            # Check if opponent has any valid moves
            self.current_player = 3 - self.current_player  # Switch player
            opponent_valid_moves = self.valid_moves()
            if not opponent_valid_moves:
                # Opponent cannot move: Current player wins
                self.done = True
                return self.board.copy(), 1, True, False, {}
            else:
                # Game continues
                return self.board.copy(), -10, False, False, {}

    def render(self):
        board_str = ""
        symbols = {0: " ", 1: "X", 2: "O", -1: "#"}
        for i in range(4):
            row_str = "|"
            for j in range(4):
                pos = i * 4 + j
                row_str += f" {symbols[self.board[pos]]} |"
            board_str += "-" * 21 + "\n" + row_str + "\n"
        board_str += "-" * 21 + "\n"
        return board_str

    def valid_moves(self):
        if self.done:
            return []
        if self.initial_placement_phase:
            # Any unoccupied cell
            return [i for i in range(16) if self.board[i] == 0]
        else:
            current_pos = self.player_positions[self.current_player]
            adjacent_indices = self.get_adjacent_indices(current_pos)
            # Return unoccupied adjacent cells
            return [idx for idx in adjacent_indices if self.board[idx] == 0]

    def get_adjacent_indices(self, position):
        row = position // 4
        col = position % 4
        adjacent_indices = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row = row + dr
                new_col = col + dc
                if 0 <= new_row < 4 and 0 <= new_col < 4:
                    adjacent_index = new_row * 4 + new_col
                    adjacent_indices.append(adjacent_index)
        return adjacent_indices
