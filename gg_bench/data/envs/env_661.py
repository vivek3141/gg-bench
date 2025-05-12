import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation spaces
        # Action space: 9 digits (1-9) * 3 positions (hundreds, tens, ones) = 27 possible actions
        self.action_space = spaces.Discrete(27)
        # Observation space: Own board (3), opponent's board (3), available digits (9) = 15-dimensional space
        self.observation_space = spaces.Box(low=0, high=9, shape=(15,), dtype=np.int64)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Player boards: index 0 for Player 1, index 1 for Player 2
        self.boards = [np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)]
        # Available digits: 1 indicates available, 0 indicates used
        self.available_digits = np.ones(9, dtype=np.int64)
        self.current_player = 0  # Player 0 starts
        self.done = False
        self.turn_counter = 0  # Total number of turns taken
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def _get_observation(self):
        own_board = self.boards[self.current_player]
        opponent_board = self.boards[1 - self.current_player]
        observation = np.concatenate([own_board, opponent_board, self.available_digits])
        return observation

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}
        # Decode action into digit and position
        digit_index = action // 3  # 0-8 corresponding to digits 1-9
        position_index = action % 3  # 0: hundreds, 1: tens, 2: ones

        digit = digit_index + 1  # Actual digit (1-9)
        position = position_index  # Position on the board

        # Check if the digit is available
        if self.available_digits[digit_index] == 0:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}
        # Check if the position on own board is empty
        own_board = self.boards[self.current_player]
        if own_board[position_index] != 0:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move
        self.available_digits[digit_index] = 0  # Mark digit as used
        own_board[position_index] = digit  # Assign digit to position

        self.turn_counter += 1
        reward = 0

        # Check if game is over (both players have filled their boards)
        if self.turn_counter >= 6:
            self.done = True
            # Compute the three-digit numbers for both players
            own_number = self._compute_number(self.boards[self.current_player])
            opponent_number = self._compute_number(self.boards[1 - self.current_player])
            # Determine the winner
            if own_number < opponent_number:
                reward = 1  # Current player wins
            else:
                reward = 0  # Current player loses
        else:
            # Switch to the next player
            self.current_player = 1 - self.current_player

        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def _compute_number(self, board):
        # Compute the three-digit number from the board positions
        number = (
            (board[0] if board[0] != 0 else 0) * 100
            + (board[1] if board[1] != 0 else 0) * 10
            + (board[2] if board[2] != 0 else 0)
        )
        return number

    def render(self):
        own_board = self.boards[self.current_player]
        opponent_board = self.boards[1 - self.current_player]
        available_digits = [i + 1 for i in range(9) if self.available_digits[i] == 1]

        result = f"Current Player: Player {self.current_player + 1}\n"
        result += (
            f"Your Board: [{self._display_digit(own_board[0])}] "
            f"[{self._display_digit(own_board[1])}] "
            f"[{self._display_digit(own_board[2])}]\n"
        )
        result += (
            f"Opponent's Board: [{self._display_digit(opponent_board[0])}] "
            f"[{self._display_digit(opponent_board[1])}] "
            f"[{self._display_digit(opponent_board[2])}]\n"
        )
        result += f"Available Digits: {' '.join(map(str, available_digits))}"
        return result

    def _display_digit(self, digit):
        return str(digit) if digit != 0 else "_"

    def valid_moves(self):
        if self.done:
            return []
        valid_actions = []
        # Available digits
        available_digit_indices = np.where(self.available_digits == 1)[0]
        # Available positions on own board
        own_board = self.boards[self.current_player]
        available_positions = np.where(own_board == 0)[0]
        # Generate valid actions
        for digit_index in available_digit_indices:
            for position_index in available_positions:
                action = digit_index * 3 + position_index
                valid_actions.append(action)
        return valid_actions
