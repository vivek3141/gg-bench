import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: numbers 1 to 9 (indices 0 to 8)
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # - observation[0]: last number played (0 if no number yet)
        # - observation[1:]: available numbers (1 if available, 0 if used)
        self.observation_space = spaces.Box(low=0, high=9, shape=(10,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sequence = []  # Sequence of numbers played
        self.available_numbers = [1] * 9  # 1 if number is available, 0 if used
        self.current_player = 1  # Players: 1 or 2
        self.last_number = 0  # Last number played (0 if no number yet)
        self.done = False

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if the current player has any valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Current player has no valid moves and loses
            self.done = True
            return self._get_observation(), -1, True, False, {}

        # Check if the action is valid
        if action not in valid_actions:
            # Invalid move: action is not valid
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move
        number = action + 1  # Convert action to number (1 to 9)

        # Update the game state
        self.available_numbers[action] = 0  # Mark number as used
        self.sequence.append(number)
        self.last_number = number

        # Switch to the other player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        # Check if the opponent has any valid moves
        opponent_valid_moves = self.valid_moves()
        if not opponent_valid_moves:
            # Opponent has no valid moves; current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Continue the game
        return self._get_observation(), 0, False, False, {}

    def render(self):
        output = f"Current Player: {self.current_player}\n"
        output += f"Sequence: {self.sequence}\n"
        available_numbers = [i + 1 for i in range(9) if self.available_numbers[i]]
        output += f"Available Numbers: {available_numbers}\n"
        return output

    def valid_moves(self):
        valid_actions = []
        for idx in range(9):
            if self.available_numbers[idx]:
                candidate_number = idx + 1
                if (
                    self.last_number == 0
                    or candidate_number % self.last_number == 0
                    or self.last_number % candidate_number == 0
                ):
                    valid_actions.append(idx)
        return valid_actions

    def _get_observation(self):
        # Observation consists of last number and available numbers
        observation = np.zeros(10, dtype=np.int32)
        observation[0] = self.last_number  # Last number played
        observation[1:] = self.available_numbers  # Available numbers
        return observation
