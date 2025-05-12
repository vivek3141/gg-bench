import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 30 possible actions (10 digits * 3 positions)
        self.action_space = spaces.Discrete(30)

        # Observation space:
        # - Player's board: 3 positions (values 0-10, where 0=empty, 1-10=digits 0-9 +1)
        # - Opponent's board: 3 positions (same encoding)
        # - Digit pool availability: 10 digits (values 0 or 1)
        self.observation_space = spaces.MultiDiscrete([11] * 6 + [2] * 10)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Digit pool: True if digit is available, False if not
        self.digit_pool = [True] * 10  # Digits 0-9
        # Player boards: List of two players, each with 3 positions initialized to -1 (empty)
        self.player_boards = [[-1, -1, -1], [-1, -1, -1]]
        # Current player (0 or 1)
        self.current_player = 0
        # Game done flag
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        # Check if game is already over
        if self.done:
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Map action index to digit and position
        digit = action // 3
        position = action % 3

        # Check if digit is available in digit pool
        if digit < 0 or digit > 9 or not self.digit_pool[digit]:
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid move

        # Check if position on current player's board is empty
        if (
            position < 0
            or position > 2
            or self.player_boards[self.current_player][position] != -1
        ):
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid move

        # Valid move: place digit on player's board and update digit pool
        self.player_boards[self.current_player][position] = digit
        self.digit_pool[digit] = False  # Digit is no longer available

        # Check if player's board is complete
        if all(pos != -1 for pos in self.player_boards[self.current_player]):
            # Form the number from the digits
            left, middle, right = self.player_boards[self.current_player]
            number = left * 100 + middle * 10 + right
            # Check if number is divisible by 7
            if number % 7 == 0:
                self.done = True
                return self._get_obs(), 1, True, False, {}  # Current player wins
        # Check if all digits have been used (game over)
        if not any(self.digit_pool):
            self.done = True
            return self._get_obs(), 0, True, False, {}  # Game over, no more digits

        # Switch to the next player
        self.current_player = 1 - self.current_player
        return self._get_obs(), 0, False, False, {}  # Continue the game

    def render(self):
        board_str = "\nDigit Dilemma Game State:\n"
        board_str += f"Current Player: Player {self.current_player + 1}\n"
        board_str += "Available Digits: "
        available_digits = [
            str(i) for i, available in enumerate(self.digit_pool) if available
        ]
        board_str += ", ".join(available_digits) + "\n"
        for i in range(2):
            player_board = self.player_boards[i]
            board_display = ["_" if pos == -1 else str(pos) for pos in player_board]
            board_str += f"Player {i + 1}'s Number: {board_display[0]} {board_display[1]} {board_display[2]}\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        # For each digit
        for digit in range(10):
            if self.digit_pool[digit]:
                # For each position
                for position in range(3):
                    if self.player_boards[self.current_player][position] == -1:
                        action = digit * 3 + position
                        valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        # Encode the observation according to the space
        # Player's board
        player_board = self.player_boards[self.current_player]
        player_board_obs = [self._encode_position(pos) for pos in player_board]
        # Opponent's board
        opponent_board = self.player_boards[1 - self.current_player]
        opponent_board_obs = [self._encode_position(pos) for pos in opponent_board]
        # Digit pool availability
        digit_pool_obs = [1 if available else 0 for available in self.digit_pool]
        # Combine into one observation vector
        obs = player_board_obs + opponent_board_obs + digit_pool_obs
        return np.array(obs, dtype=np.int32)

    def _encode_position(self, pos):
        if pos == -1:
            return 0  # Empty position
        else:
            return pos + 1  # Digit 0-9 becomes 1-10
