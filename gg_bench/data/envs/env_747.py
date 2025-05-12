import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are three possible actions: select digit at position 0, 1, or 2 (hundreds, tens, ones)
        self.action_space = spaces.Discrete(3)

        # Observation space consists of the digits of the current player's number and the opponent's number
        # Each digit ranges from 0 to 9
        self.observation_space = spaces.Box(low=0, high=9, shape=(6,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Each player's number is represented as a list of digits [hundreds, tens, ones]
        self.player_numbers = [
            [9, 9, 9],  # Player 0's digits
            [9, 9, 9],  # Player 1's digits
        ]
        # Current player: 0 or 1
        self.current_player = 0
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return self._get_obs(), -10, True, False, {}

        # Check if current player has any valid moves
        if not self._player_has_valid_moves(self.current_player):
            # Current player must pass; switch to opponent
            self.current_player = 1 - self.current_player
            return (
                self._get_obs(),
                0,
                False,
                False,
                {"message": "No valid moves, turn passed."},
            )

        # Validate action
        if action not in [0, 1, 2]:
            # Invalid action index
            self.done = True
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {"message": "Invalid action index."},
            )

        own_digits = self.player_numbers[self.current_player]

        if own_digits[action] <= 0:
            # Invalid move: selected digit is zero or less
            self.done = True
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {"message": "Invalid move: selected digit is zero."},
            )
        else:
            # Valid move
            digit_value = own_digits[action]

            # Subtract digit value from opponent's number
            opponent = 1 - self.current_player
            opponent_digits = self.player_numbers[opponent]
            opponent_number = (
                opponent_digits[0] * 100 + opponent_digits[1] * 10 + opponent_digits[2]
            )
            new_opponent_number = max(0, opponent_number - digit_value)

            # Update opponent's digits
            new_opponent_digits = [
                new_opponent_number // 100,
                (new_opponent_number % 100) // 10,
                new_opponent_number % 10,
            ]
            self.player_numbers[opponent] = new_opponent_digits

            # Decrement own digit by 1
            own_digits[action] -= 1

            # Check if opponent's number has reached zero
            if sum(self.player_numbers[opponent]) == 0:
                # Current player wins
                self.done = True
                return (
                    self._get_obs(),
                    1,
                    True,
                    False,
                    {"message": "Current player wins!"},
                )
            else:
                # Switch to opponent
                self.current_player = opponent
                return self._get_obs(), 0, False, False, {}

    def render(self):
        # Return a visual representation of the environment state as a string
        own_digits = self.player_numbers[self.current_player]
        opponent_digits = self.player_numbers[1 - self.current_player]
        representation = (
            f"Current Player: Player {self.current_player + 1}\n"
            f"Your Number: {own_digits[0]} {own_digits[1]} {own_digits[2]}\n"
            f"Opponent's Number: {opponent_digits[0]} {opponent_digits[1]} {opponent_digits[2]}"
        )
        return representation

    def valid_moves(self):
        # Return a list of valid action indices (0, 1, 2) where the current player's digits are greater than zero
        own_digits = self.player_numbers[self.current_player]
        return [i for i in range(3) if own_digits[i] > 0]

    def _player_has_valid_moves(self, player_index):
        own_digits = self.player_numbers[player_index]
        return any(digit > 0 for digit in own_digits)

    def _get_obs(self):
        own_digits = self.player_numbers[self.current_player]
        opponent_digits = self.player_numbers[1 - self.current_player]
        # Observation is an array of own digits followed by opponent's digits
        observation = np.array(own_digits + opponent_digits, dtype=np.int32)
        return observation
