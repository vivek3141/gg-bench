import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete actions from 0 to 48 corresponding to numbers 2 to 50
        self.action_space = spaces.Discrete(49)

        # Observation space: Box containing availability, last number selected, and current player
        # First 49 elements: availability of numbers 2 to 50 (0 if available, 1 if selected)
        # Element 49: normalized last number selected (-1.0 indicates no number selected yet)
        # Element 50: current player's turn (1 or -1)
        low = np.array([0.0] * 49 + [-1.0] + [-1.0], dtype=np.float32)
        high = np.array([1.0] * 49 + [1.0] + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the availability of numbers and game state
        self.availability = np.zeros(
            49, dtype=np.float32
        )  # Numbers 2-50 are all available at the start
        self.last_number = -1.0  # No last number selected yet
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return initial observation and empty info

    def step(self, action):
        if self.done:
            # Game is over, no further actions can be taken
            return self._get_observation(), 0, self.done, False, {}

        # Validate action
        if action < 0 or action >= 49:
            # Action is out of bounds
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, {}

        number = action + 2  # Map action index to number in the pool (2-50)

        # Check if the number has already been selected
        if self.availability[action] == 1.0:
            # Number has been previously selected
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, {}

        # Check if the move is valid
        if self.last_number == -1.0:
            # First move of the game; any available number is valid
            is_valid = True
        else:
            last_number = self._denormalize_number(self.last_number)
            # Move is valid if the number is a divisor or multiple of the last number selected
            is_valid = number % last_number == 0 or last_number % number == 0

        if not is_valid:
            # Move is invalid
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, {}

        # Move is valid; update game state
        self.availability[action] = 1.0
        self.last_number = self._normalize_number(number)

        # Check if the next player has any valid moves
        self.current_player *= -1  # Switch to the other player
        if not self._has_valid_moves():
            # Opponent cannot make a valid move; current player wins
            self.done = True
            reward = 1  # Winning reward
        else:
            reward = 0  # No reward; game continues

        observation = self._get_observation()
        return (
            observation,
            reward,
            self.done,
            False,
            {},
        )  # Return observation, reward, done, truncated, info

    def render(self):
        # Generate a string representation of the game state
        available_numbers = [str(i + 2) for i in range(49) if self.availability[i] == 0]
        state = f"Available Numbers: {', '.join(available_numbers)}"
        if self.last_number == -1.0:
            last_number_str = "None"
        else:
            last_number_str = str(self._denormalize_number(self.last_number))
        current_player_str = "Player 1" if self.current_player == 1 else "Player 2"
        return (
            f"{state}\n"
            f"Last Number Selected: {last_number_str}\n"
            f"Current Player: {current_player_str}"
        )

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        if self.done:
            return []

        if self.last_number == -1.0:
            # First move: all available numbers are valid
            valid_actions = [i for i in range(49) if self.availability[i] == 0]
        else:
            last_number = self._denormalize_number(self.last_number)
            valid_actions = []
            for i in range(49):
                if self.availability[i] == 0:
                    number = i + 2
                    if number % last_number == 0 or last_number % number == 0:
                        valid_actions.append(i)
        return valid_actions

    def _has_valid_moves(self):
        # Check if the current player has any valid moves
        return len(self.valid_moves()) > 0

    def _normalize_number(self, number):
        # Normalize the number to a value between 0.0 and 1.0
        return (number - 2) / 48.0

    def _denormalize_number(self, normalized_number):
        # Convert the normalized number back to an integer in the range 2-50
        return int(round(normalized_number * 48.0 + 2))

    def _get_observation(self):
        # Construct the observation array
        observation = np.concatenate(
            (
                self.availability.copy(),
                np.array([self.last_number], dtype=np.float32),
                np.array([self.current_player], dtype=np.float32),
            )
        )
        return observation.astype(np.float32)
