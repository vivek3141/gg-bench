import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: digits 1-9 mapped to indices 0-8
        self.action_space = spaces.Discrete(9)

        # Observation space: an array of 9 elements
        # 0: digit is available
        # 1: selected by current player
        # -1: selected by opponent
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Available digits: digits 1-9, mapped to indices 0-8
        self.available_digits = list(range(1, 10))

        # Player selections
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.player1_digits = []
        self.player2_digits = []

        # Initialize the board
        self.board = np.zeros(9, dtype=np.int8)

        self.done = False

        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        digit = action + 1  # Map action index to digit (1-9)

        # Check if digit is available
        if self.board[action] != 0 or digit not in self.available_digits:
            # Invalid move: digit not available
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Check rule: cannot select a digit that sums to 10 with any opponent's selected digit
        opponent_digits = (
            self.player2_digits if self.current_player == 1 else self.player1_digits
        )

        if any(digit + opp_digit == 10 for opp_digit in opponent_digits):
            # Invalid move according to game rules
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Valid move, update state
        self.board[action] = self.current_player
        self.available_digits.remove(digit)
        if self.current_player == 1:
            self.player1_digits.append(digit)
        else:
            self.player2_digits.append(digit)

        # Now, check if the game is over (opponent has to pick last digit and loses)
        next_player = -self.current_player
        opponent_digits = (
            self.player1_digits if next_player == 1 else self.player2_digits
        )

        # Generate list of valid moves for opponent
        valid_moves = self.get_valid_moves(next_player, opponent_digits)

        if not valid_moves:
            # Opponent has no valid moves, forced to select last digit and loses
            self.done = True
            # Current player wins
            return self.board.copy(), 1, True, False, {}

        # Switch player
        self.current_player = next_player

        # Continue game
        return self.board.copy(), 0, False, False, {}

    def get_valid_moves(self, player, opponent_digits):
        valid_moves = []
        for action in range(9):
            digit = action + 1
            if self.board[action] != 0:
                continue  # Already taken
            # Check if digit sums to 10 with any of opponent's digits
            if any(digit + opp_digit == 10 for opp_digit in opponent_digits):
                continue  # Can't select this digit
            valid_moves.append(action)
        return valid_moves

    def valid_moves(self):
        # Return list of valid action indices for current player
        opponent_digits = (
            self.player2_digits if self.current_player == 1 else self.player1_digits
        )
        return self.get_valid_moves(self.current_player, opponent_digits)

    def render(self):
        available = [str(i + 1) for i in range(9) if self.board[i] == 0]
        player1_picks = [str(d) for d in self.player1_digits]
        player2_picks = [str(d) for d in self.player2_digits]

        board_repr = f"Available Digits: {' '.join(available)}\n"
        board_repr += f"Player 1's picks: {', '.join(player1_picks)}\n"
        board_repr += f"Player 2's picks: {', '.join(player2_picks)}\n"
        board_repr += f"Current player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"

        return board_repr
