import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: numbers from 1 to 100 (indices 0 to 99)
        self.action_space = spaces.Discrete(100)

        # Observation space: 'last_number' and 'used_numbers'
        self.observation_space = spaces.Dict(
            {
                "last_number": spaces.Discrete(101),  # 0 (no last number) to 100
                "used_numbers": spaces.MultiBinary(100),  # numbers 1 to 100
            }
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.last_number = 0  # No last number at start
        self.used_numbers = np.zeros(100, dtype=np.int8)  # Numbers 1..100
        self.done = False

        return self._get_observation(), {}

    def _get_observation(self):
        return {
            "last_number": self.last_number,
            "used_numbers": self.used_numbers.copy(),
        }

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Convert action to selected number
        selected_number = action + 1  # actions are 0..99, numbers are 1..100

        # Check if selected_number has been used
        if self.used_numbers[selected_number - 1]:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # If last_number == 0, it's the first move
        if self.last_number == 0:
            # Any number is valid
            valid = True
        else:
            # Check if selected_number is a divisor or multiple of last_number, and not equal
            if selected_number == self.last_number:
                valid = False
            elif (
                self.last_number % selected_number == 0
                or selected_number % self.last_number == 0
            ):
                valid = True
            else:
                valid = False

        if not valid:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move
        self.used_numbers[selected_number - 1] = 1  # Mark number as used
        self.last_number = selected_number  # Update last_number

        # Now check if the opponent has any valid moves
        opponent_valid_moves = self.valid_moves()

        if not opponent_valid_moves:
            # Opponent cannot move, current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        else:
            # Game continues
            # The agent needs to play as the opponent in the next step
            return self._get_observation(), 0, False, False, {}

    def valid_moves(self):
        # Returns list of valid actions (indices 0..99)
        if self.last_number == 0:
            # First move, all numbers 1..100 are valid (if not used)
            return [i for i in range(100) if self.used_numbers[i] == 0]
        else:
            moves = []
            for i in range(100):
                num = i + 1
                if self.used_numbers[i]:
                    continue  # Already used
                if num == self.last_number:
                    continue  # Cannot repeat last number
                if self.last_number % num == 0 or num % self.last_number == 0:
                    moves.append(i)
            return moves

    def render(self):
        # Return a string representation of the sequence
        sequence = [i + 1 for i in range(100) if self.used_numbers[i]]
        return "Sequence: " + " -> ".join(map(str, sequence))
