import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Precompute all arithmetic sequences of length 3 from numbers 1 to 9
        self.arithmetic_sequences = [
            {1, 2, 3},
            {2, 3, 4},
            {3, 4, 5},
            {4, 5, 6},
            {5, 6, 7},
            {6, 7, 8},
            {7, 8, 9},  # Common difference d=1
            {1, 3, 5},
            {2, 4, 6},
            {3, 5, 7},
            {4, 6, 8},
            {5, 7, 9},  # Common difference d=2
            {1, 4, 7},
            {2, 5, 8},
            {3, 6, 9},  # Common difference d=3
            {1, 5, 9},  # Common difference d=4
            {3, 5, 7},  # Common difference d=2 (reverse)
            {1, 5, 9},  # Common difference d=4
        ]

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if action not in self.action_space or self.board[action] != 0 or self.done:
            # Invalid move
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Update the board
        self.board[action] = self.current_player

        # Get current player's numbers
        current_player_numbers = set(np.where(self.board == self.current_player)[0] + 1)

        # Check if current player loses
        if self._has_arithmetic_sequence(current_player_numbers):
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Get opponent's numbers
        opponent_numbers = set(np.where(self.board == -self.current_player)[0] + 1)

        # Check if current player wins (opponent loses)
        if self._has_arithmetic_sequence(opponent_numbers):
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Check for draw (no more available moves)
        if np.all(self.board != 0):
            self.done = True
            return self.board.copy(), 0, True, False, {}

        # Switch player
        self.current_player *= -1

        return self.board.copy(), 0, False, False, {}

    def render(self):
        available_numbers = [str(i + 1) for i in range(9) if self.board[i] == 0]
        player1_numbers = [str(i + 1) for i in range(9) if self.board[i] == 1]
        player2_numbers = [str(i + 1) for i in range(9) if self.board[i] == -1]

        state_str = f"Available Numbers: {' '.join(available_numbers)}\n"
        state_str += f"Player 1's Numbers: {' '.join(player1_numbers)}\n"
        state_str += f"Player 2's Numbers: {' '.join(player2_numbers)}\n"
        return state_str

    def valid_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def _has_arithmetic_sequence(self, numbers):
        if len(numbers) < 3:
            return False
        for sequence in self.arithmetic_sequences:
            if sequence.issubset(numbers):
                return True
        return False
