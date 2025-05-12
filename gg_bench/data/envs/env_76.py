import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=123456):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(10)  # Digits 0-9
        self.MAX_DIGITS = 20  # Maximum number of digits in the number
        self.observation_space = spaces.Box(
            low=-1, high=9, shape=(self.MAX_DIGITS,), dtype=np.int32
        )

        self.starting_number = starting_number
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the shared number as a list of digits
        digits = [int(d) for d in str(self.starting_number)]
        self.shared_number = digits + [-1] * (self.MAX_DIGITS - len(digits))
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.shared_number, {}  # Observation and info

    def step(self, action):
        if self.done:
            return self.shared_number, 0, True, False, {}

        if self.is_zero():
            # No valid moves for current player
            self.done = True
            reward = -1  # Current player loses
            return self.shared_number, reward, True, False, {}

        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid action
            self.done = True
            reward = -10  # Penalty for invalid move
            return self.shared_number, reward, True, False, {}

        # Remove the first occurrence of the digit
        idx = next((i for i, d in enumerate(self.shared_number) if d == action), None)
        if idx is not None:
            self.shared_number.pop(idx)
            self.shared_number.append(-1)  # Keep the length constant
        else:
            # Should not happen as we've checked valid moves
            self.done = True
            reward = -10
            return self.shared_number, reward, True, False, {}

        # Remove leading zeros
        while self.shared_number and self.shared_number[0] == 0:
            self.shared_number.pop(0)
            self.shared_number.append(-1)

        # Check if number is reduced to zero after the move
        if self.is_zero():
            self.done = True
            reward = -1  # Current player loses
            return self.shared_number, reward, True, False, {}

        # Check if opponent has any valid moves
        opponent_valid_moves = self.valid_moves()
        if not opponent_valid_moves:
            # Opponent has no valid moves, current player wins
            self.done = True
            reward = 1  # Current player wins
            return self.shared_number, reward, True, False, {}

        # Switch to next player
        self.current_player *= -1

        # Game continues
        reward = -10  # Penalty for valid move
        return self.shared_number, reward, False, False, {}

    def valid_moves(self):
        if self.is_zero():
            return []
        digits = [d for d in self.shared_number if d != -1]
        return list(set(digits))

    def is_zero(self):
        digits = [d for d in self.shared_number if d != -1]
        return len(digits) == 0

    def render(self):
        digits = [d for d in self.shared_number if d != -1]
        if digits:
            number_str = "".join(map(str, digits))
        else:
            number_str = "0"
        player_str = "Player 1" if self.current_player == 1 else "Player 2"
        return f"Current Number: {number_str}\nCurrent Player: {player_str}"
