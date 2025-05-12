import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: digits 1 to 9 (indices 0-8)
        self.action_space = spaces.Discrete(9)

        # Define observation space: sequence and last_digit
        # sequence: positions 0-8, values 0 or 1
        # last_digit: position 9, values -1 to 9 (-1 indicates no digit has been removed yet)
        low = np.array([0] * 9 + [-1], dtype=np.int8)
        high = np.array([1] * 9 + [9], dtype=np.int8)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the sequence to all digits available
        self.sequence = np.ones(9, dtype=np.int8)

        # No digit has been removed yet
        self.last_digit = -1

        # Player 1 starts (can be 1 or -1)
        self.current_player = 1

        self.done = False

        # Return the initial observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}  # Game already over

        # Map action to digit (1 to 9)
        digit = action + 1  # action is 0-8, digit is 1-9

        # Check if digit is available
        if self.sequence[action] == 0:
            # Invalid move, digit already removed
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if move is valid
        if self.last_digit == -1:
            # First move, any digit can be removed
            valid_move = True
        else:
            # Must remove a digit that is a divisor or multiple of last_digit
            if digit % self.last_digit == 0 or self.last_digit % digit == 0:
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move, remove digit
        self.sequence[action] = 0
        self.last_digit = digit

        # Check if opponent has any valid moves
        opponent_valid_moves = self._get_valid_moves()

        if len(opponent_valid_moves) == 0:
            # Opponent cannot move, current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}
        else:
            # Switch to next player
            self.current_player *= -1
            return self._get_observation(), 0, False, False, {}

    def _get_observation(self):
        # Observation is an array of length 10
        # First 9 positions are sequence, last position is last_digit
        observation = np.concatenate(
            (self.sequence.copy(), np.array([self.last_digit], dtype=np.int8))
        )
        return observation

    def render(self):
        # Build the sequence string
        sequence_str = " ".join(
            [str(i + 1) if self.sequence[i] == 1 else " " for i in range(9)]
        )
        last_digit_str = f"{self.last_digit}" if self.last_digit != -1 else "None"
        print(f"Current sequence:\n{sequence_str}")
        print(f"Last digit removed: {last_digit_str}\n")

    def valid_moves(self):
        # Return list of valid actions (indices 0-8)
        return self._get_valid_moves()

    def _get_valid_moves(self):
        valid_moves = []
        if self.last_digit == -1:
            # First move, all available digits are valid
            for i in range(9):
                if self.sequence[i] == 1:
                    valid_moves.append(i)
        else:
            # Must pick a digit that is a divisor or multiple of last_digit
            for i in range(9):
                if self.sequence[i] == 1:
                    digit = i + 1  # digits 1 to 9
                    if digit % self.last_digit == 0 or self.last_digit % digit == 0:
                        valid_moves.append(i)
        return valid_moves
