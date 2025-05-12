import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 8 tokens * 16 cells = 128 possible actions
        self.action_space = spaces.Discrete(128)

        # Observation space: 16 cells with values from -8 to 8 (-8 to -1 for Player 2, 0 for empty, 1 to 8 for Player 1)
        self.observation_space = spaces.Box(low=-8, high=8, shape=(16,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(16, dtype=np.int8)  # 4x4 grid flattened to 16 cells
        self.current_player = 1  # Player 1 starts (1 for Player 1, -1 for Player 2)
        self.done = False

        # Tokens numbered 1 to 8 for each player
        self.unplaced_tokens = {
            1: set(range(1, 9)),  # Player 1's unplaced tokens
            -1: set(range(1, 9)),  # Player 2's unplaced tokens
        }
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Decode action
        token_number = (action // 16) + 1  # Tokens are numbered from 1 to 8
        cell_index = action % 16  # Cell indices from 0 to 15

        # Check if token_number is in current player's unplaced tokens
        if token_number not in self.unplaced_tokens[self.current_player]:
            return self._get_obs(), -10, True, False, {}

        # Check if the selected cell is empty
        if self.board[cell_index] != 0:
            return self._get_obs(), -10, True, False, {}

        # Place the token
        self.board[cell_index] = token_number * self.current_player
        self.unplaced_tokens[self.current_player].remove(token_number)

        # Check for victory condition
        if self._check_victory(cell_index):
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Check for draw (grid is full)
        if np.all(self.board != 0):
            self.done = True
            # Calculate the total sums for both players
            player1_sum = np.sum(self.board[self.board > 0])
            player2_sum = -np.sum(self.board[self.board < 0])
            if player1_sum > player2_sum:
                reward = 1 if self.current_player == 1 else -1
            elif player2_sum > player1_sum:
                reward = 1 if self.current_player == -1 else -1
            else:
                # If sums are equal, last player to place a token wins
                reward = 1
            return self._get_obs(), reward, True, False, {}

        # Switch players
        self.current_player *= -1
        return self._get_obs(), 0, False, False, {}

    def render(self):
        # Create a visual representation of the board
        symbols = {0: "    "}
        for player in [1, -1]:
            for num in range(1, 9):
                token = f"P{1 if player == 1 else 2}-{num}"
                symbols[player * num] = f"{token: <4}"

        rows = ["A", "B", "C", "D"]
        cols = ["1", "2", "3", "4"]
        line = "+" + "----+" * 4

        board_str = "     " + "   ".join(cols) + "\n" + line + "\n"
        for i in range(4):
            row_cells = "|".join(symbols[self.board[i * 4 + j]] for j in range(4))
            board_str += f"{rows[i]}|{row_cells}|\n" + line + "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        for token_number in self.unplaced_tokens[self.current_player]:
            for cell_index in range(16):
                if self.board[cell_index] == 0:
                    action = (token_number - 1) * 16 + cell_index
                    valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        return self.board.copy()

    def _check_victory(self, last_move_index):
        # Get the position of the last move
        row = last_move_index // 4
        col = last_move_index % 4

        # Directions: horizontal, vertical, two diagonals
        directions = [
            ((0, -1), (0, 1)),  # Horizontal
            ((-1, 0), (1, 0)),  # Vertical
            ((-1, -1), (1, 1)),  # Main diagonal
            ((-1, 1), (1, -1)),  # Anti-diagonal
        ]

        for direction in directions:
            sequence = self._collect_sequence(row, col, direction)
            if len(sequence) >= 3:
                # Check for strictly ascending or descending order
                numbers = [abs(self.board[r * 4 + c]) for r, c in sequence]
                if self._is_strict_sequence(numbers):
                    return True
        return False

    def _collect_sequence(self, row, col, direction):
        sequence = [(row, col)]
        player = self.current_player
        for delta in direction:
            r, c = row, col
            while True:
                r += delta[0]
                c += delta[1]
                if 0 <= r < 4 and 0 <= c < 4 and self.board[r * 4 + c] * player > 0:
                    sequence.append((r, c))
                else:
                    break
        return sequence

    def _is_strict_sequence(self, numbers):
        if len(numbers) < 3:
            return False
        ascending = all(x < y for x, y in zip(numbers, numbers[1:]))
        descending = all(x > y for x, y in zip(numbers, numbers[1:]))
        return ascending or descending
