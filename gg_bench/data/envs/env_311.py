import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: digits 1-9 placed at 'front' or 'back' => 9 digits * 2 positions = 18 actions
        self.action_space = spaces.Discrete(18)

        # Observation space: Both players' numbers, up to 7 digits each
        # Represented as arrays of digits, padded with zeros (0 indicates empty positions)
        # Shape: (2, 7) -> 2 players, 7-digit numbers
        self.observation_space = spaces.Box(low=0, high=9, shape=(2, 7), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize players' numbers as empty arrays
        self.numbers = np.zeros((2, 7), dtype=np.int8)
        # Lengths of each player's number
        self.lengths = [0, 0]
        # Current player (0 or 1)
        self.current_player = 0
        # Game over flag
        self.done = False
        return self._get_observation(), {}  # Observation and info

    def _get_observation(self):
        # Return a copy of the current game state
        return self.numbers.copy()

    def _decode_action(self, action):
        # Map action index to (digit, position)
        # Actions 0-17 correspond to digits 1-9 and positions 'front' or 'back'
        digit = (action // 2) + 1  # Digits from 1 to 9
        position = "front" if action % 2 == 0 else "back"
        return digit, position

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        if self.done:
            return []
        if self.lengths[self.current_player] >= 7:
            return []
        else:
            return list(
                range(18)
            )  # All actions are valid if number length is less than 7

    def _is_palindrome(self, digits):
        # Check if the array of digits forms a palindrome
        return np.array_equal(digits, digits[::-1])

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}
        # Decode the action into digit and position
        digit, position = self._decode_action(action)
        player = self.current_player
        length = self.lengths[player]
        if length >= 7:
            # Cannot add more digits, invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}
        # Add the digit to the player's number
        if position == "front":
            # Shift existing digits to the right and insert the new digit at the front
            self.numbers[player, 1 : length + 1] = self.numbers[player, 0:length]
            self.numbers[player, 0] = digit
        elif position == "back":
            # Insert the new digit at the back
            self.numbers[player, length] = digit
        self.lengths[player] += 1
        # Check if the player has formed a palindrome of at least three digits
        number_digits = self.numbers[player, 0 : self.lengths[player]]
        if self.lengths[player] >= 3 and self._is_palindrome(number_digits):
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}
        # Check for a draw if both players have reached the maximum length
        if self.lengths[0] >= 7 and self.lengths[1] >= 7:
            # Game is a draw
            self.done = True
            reward = 0
            return self._get_observation(), reward, True, False, {}
        # Switch to the other player
        self.current_player = 1 - self.current_player
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        # Create a string representation of the current game state
        output = ""
        for i in range(2):
            number_digits = self.numbers[i, 0 : self.lengths[i]]
            number_str = "".join(map(str, number_digits))
            player_str = f"Player {i + 1}'s number: {number_str}\n"
            output += player_str
        return output
