import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 cells and an extra action for passing (action 9)
        self.action_space = spaces.Discrete(10)

        # The observation will be the state of the grid, flattened into a 1D array
        # 0: empty, 1: Player X, -1: Player O
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # 1 for Player X, -1 for Player O
        self.opponent_last_move = None  # Index of opponent's last move
        self.last_player_passed = False
        self.last_player_to_move = None  # Tracks the last player who made a valid move
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        reward = 0

        valid_moves = self.valid_moves()

        # Action 9 is pass
        if action == 9:
            if len(valid_moves) > 0:
                # Passing when you have valid moves is invalid
                self.done = True
                return self.board.copy(), -10, True, False, {}
            else:
                if self.last_player_passed:
                    # Both players passed consecutively, game ends
                    self.done = True
                    if self.last_player_to_move == self.current_player * -1:
                        # Opponent was the last to make a move, so they win
                        reward = -1
                    else:
                        # Current player wins (since opponent didn't make a move)
                        reward = 1
                    return self.board.copy(), reward, True, False, {}
                else:
                    # Current player passes
                    self.last_player_passed = True
                    self.current_player *= -1
                    return self.board.copy(), 0, False, False, {}
        else:
            # Attempting to make a move
            if action not in valid_moves:
                # Invalid move
                self.done = True
                return self.board.copy(), -10, True, False, {}

            # Make the move
            self.board[action] = self.current_player
            self.opponent_last_move = action
            self.last_player_passed = False
            self.last_player_to_move = self.current_player

            # Check if the opponent has any valid moves
            self.current_player *= -1
            opponent_valid_moves = self.valid_moves()
            if len(opponent_valid_moves) == 0:
                # Opponent has no valid moves, current player wins
                self.done = True
                reward = 1  # Current player wins
                self.current_player *= -1  # Switch back to current player for rendering
                return self.board.copy(), reward, True, False, {}
            else:
                # Continue the game
                self.current_player *= -1  # Switch back to opponent's turn
                return self.board.copy(), 0, False, False, {}

    def render(self):
        symbols = {1: "X", -1: "O", 0: "-"}
        board_str = ""
        for i in range(3):
            row = ""
            for j in range(3):
                index = i * 3 + j
                row += f" {symbols[self.board[index]]} "
            board_str += row.strip() + "\n"
        return board_str.strip()

    def valid_moves(self):
        # If opponent didn't make a move yet, all empty cells are valid
        if self.opponent_last_move is None:
            return [i for i in range(9) if self.board[i] == 0]

        # Get the position of the opponent's last move
        opp_row = self.opponent_last_move // 3
        opp_col = self.opponent_last_move % 3

        # Compute the set of cells adjacent to opponent's last move
        adjacent_cells = set()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r = opp_row + dr
                c = opp_col + dc
                if 0 <= r < 3 and 0 <= c < 3:
                    adjacent_cells.add(r * 3 + c)

        # Valid moves are empty cells that are not adjacent to opponent's last move
        valid_moves = [
            i for i in range(9) if self.board[i] == 0 and i not in adjacent_cells
        ]

        return valid_moves
