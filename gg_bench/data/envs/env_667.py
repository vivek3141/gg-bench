import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(30)  # Actions from 0 to 29
        self.observation_space = spaces.Box(
            low=np.array([1, 0, 0]), high=np.array([9, 9, 9]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1
        self.history = []

        # Generate a random valid starting number
        while True:
            hundreds = random.randint(1, 9)
            tens = random.randint(0, 9)
            units = random.choice([1, 3, 7, 9])  # To ensure the number is odd
            number = hundreds * 100 + tens * 10 + units
            if self.is_prime(number):
                break
        self.current_number = number
        self.digits = np.array([hundreds, tens, units], dtype=np.int32)
        self.history.append(self.current_number)
        self.done = False
        return self.digits.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.digits.copy(), 0, True, False, {}  # Game already over

        # Map action to (position, new_digit)
        position = action // 10  # 0 for hundreds, 1 for tens, 2 for units
        new_digit = action % 10

        # Copy the current digits
        digits = self.digits.copy()

        # Validate the action
        current_digit = digits[position]

        # Cannot change to the same digit
        if new_digit == current_digit:
            return self.digits.copy(), 0, True, False, {}  # Invalid action, game over

        # Hundreds place cannot be 0 and must be 1-9
        if position == 0:
            if new_digit == 0 or not (1 <= new_digit <= 9):
                return (
                    self.digits.copy(),
                    0,
                    True,
                    False,
                    {},
                )  # Invalid action, game over

        # Tens and units place digits must be 0-9
        if position in [1, 2]:
            if not (0 <= new_digit <= 9):
                return (
                    self.digits.copy(),
                    0,
                    True,
                    False,
                    {},
                )  # Invalid action, game over

        # Apply the action
        digits[position] = new_digit

        # Ensure hundreds digit is not 0 (leading zero not allowed)
        if digits[0] == 0:
            return self.digits.copy(), 0, True, False, {}  # Invalid move, game over

        # Form the new number
        new_number = digits[0] * 100 + digits[1] * 10 + digits[2]

        # Validate the new number
        if new_number in self.history:
            return (
                self.digits.copy(),
                0,
                True,
                False,
                {},
            )  # Number already used, game over
        if not self.is_prime(new_number):
            return (
                self.digits.copy(),
                0,
                True,
                False,
                {},
            )  # Not a prime number, game over
        if new_number % 2 == 0:
            return (
                self.digits.copy(),
                0,
                True,
                False,
                {},
            )  # Not an odd number, game over

        # Move is valid; update the state
        self.current_number = new_number
        self.digits = digits
        self.history.append(new_number)

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if opponent has any valid moves
        opponent_valid_moves = self.valid_moves()
        if not opponent_valid_moves:
            # Current player wins
            self.done = True
            return self.digits.copy(), 1, True, False, {}  # Win with reward 1
        else:
            # Game continues
            return (
                self.digits.copy(),
                -10,
                False,
                False,
                {},
            )  # Valid move with reward -10

    def render(self):
        return (
            f"Current number: {self.current_number}\n"
            f"Digits: {self.digits.tolist()}\n"
            f"Current player: Player {self.current_player}\n"
            f"History: {self.history}\n"
        )

    def valid_moves(self):
        # Return list of valid action indices from the current state
        valid_actions = []
        for position in range(3):
            current_digit = self.digits[position]
            if position == 0:
                possible_digits = [d for d in range(1, 10) if d != current_digit]
            else:
                possible_digits = [d for d in range(0, 10) if d != current_digit]
            for new_digit in possible_digits:
                digits = self.digits.copy()
                digits[position] = new_digit

                # Ensure hundreds digit is not zero
                if digits[0] == 0:
                    continue  # Leading zero not allowed

                new_number = digits[0] * 100 + digits[1] * 10 + digits[2]
                if new_number in self.history:
                    continue  # Number already used
                if new_number % 2 == 0:
                    continue  # Must be odd
                if not self.is_prime(new_number):
                    continue  # Must be prime

                action = position * 10 + new_digit
                valid_actions.append(action)
        return valid_actions

    @staticmethod
    def is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
