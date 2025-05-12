import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Integers from 0 to 48, representing numbers 2 to 50
        self.action_space = spaces.Discrete(49)

        # Observation space: An array of size 50
        # Elements 0-48: 1 if number (index + 2) is available, 0 if not
        # Element 49: Last number selected by the opponent, normalized between 0 and 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(50,), dtype=np.float32
        )

        # Initialize the game state
        self.available_numbers = np.ones(49, dtype=np.float32)
        self.last_number_selected = 0
        self.current_player = 1  # Player 1 starts

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(49, dtype=np.float32)
        self.last_number_selected = 0
        self.current_player = 1
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        selected_number = action + 2  # Map action to number between 2 and 50

        # Check if the selected number is available
        if self.available_numbers[action] == 0:
            # Invalid move: Number already selected
            return self._get_observation(), -10, True, False, {}

        # Check if the move is valid based on the game rules
        if self.last_number_selected != 0:
            if not (
                selected_number % self.last_number_selected == 0
                or self.last_number_selected % selected_number == 0
            ):
                # Invalid move: Not a factor or multiple
                return self._get_observation(), -10, True, False, {}

        # Valid move: Update the game state
        self.available_numbers[action] = 0  # Remove the number from the pool
        self.last_number_selected = selected_number  # Update the last selected number
        self.current_player *= -1  # Switch to the next player

        # Check if the next player has any valid moves
        if not self._has_valid_moves():
            # Current player wins
            return self._get_observation(), 1, True, False, {}

        # Game continues
        observation = self._get_observation()
        return observation, 0, False, False, {}

    def render(self):
        available_numbers = [
            str(i + 2) for i in range(49) if self.available_numbers[i] == 1
        ]
        available_numbers_str = ", ".join(available_numbers)
        return f"Available Numbers: {available_numbers_str}\nLast Number Selected by Opponent: {self.last_number_selected}"

    def valid_moves(self):
        valid_moves = []
        for i in range(49):
            if self.available_numbers[i] == 1:
                number = i + 2
                if self.last_number_selected == 0:
                    # First move: Any number is valid
                    valid_moves.append(i)
                else:
                    if (
                        number % self.last_number_selected == 0
                        or self.last_number_selected % number == 0
                    ):
                        valid_moves.append(i)
        return valid_moves

    def _get_observation(self):
        observation = np.zeros(50, dtype=np.float32)
        observation[:49] = self.available_numbers  # Numbers availability
        observation[49] = self.last_number_selected / 50.0  # Normalize last number
        return observation

    def _has_valid_moves(self):
        for i in range(49):
            if self.available_numbers[i] == 1:
                number = i + 2
                if self.last_number_selected == 0:
                    # First move: Any number is valid
                    return True
                else:
                    if (
                        number % self.last_number_selected == 0
                        or self.last_number_selected % number == 0
                    ):
                        return True
        return False
