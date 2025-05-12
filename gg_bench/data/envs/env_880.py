import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: integers from 0 to 19 (representing numbers 1 to 20)
        self.action_space = spaces.Discrete(20)

        # Observation space: array of 21 elements
        # First 20 elements represent the availability of numbers 1 to 20 (1: available, 0: removed)
        # Last element represents parity requirement: -1 (none), 0 (even), 1 (odd)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(21,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            20, dtype=np.int32
        )  # Numbers 1 to 20 are available
        self.parity_requirement = -1  # No parity requirement at the start
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        info = {}
        return observation, info  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}
        # Map action to selected number (actions are 0-indexed)
        selected_number = action + 1

        # Check if action is valid
        if not (0 <= action < 20):
            # Invalid action index
            return self._get_observation(), -10, True, False, {}

        if self.available_numbers[action] == 0:
            # Number already removed
            return self._get_observation(), -10, True, False, {}

        if self.parity_requirement != -1:
            if self.parity_requirement == 0 and selected_number % 2 != 0:
                # Parity requirement is even, but selected number is odd
                return self._get_observation(), -10, True, False, {}
            if self.parity_requirement == 1 and selected_number % 2 != 1:
                # Parity requirement is odd, but selected number is even
                return self._get_observation(), -10, True, False, {}

        # Valid move: remove the selected number
        self.available_numbers[action] = 0

        # Announce the number removed and its parity
        number_parity = "Even" if selected_number % 2 == 0 else "Odd"

        # Indicate the parity requirement for the next player
        if selected_number % 2 == 0:
            self.parity_requirement = 1  # Next player must remove an odd number
        else:
            self.parity_requirement = 0  # Next player must remove an even number

        # Check if the opponent can make a valid move
        opponent_can_move = self._check_opponent_moves()

        if not opponent_can_move:
            # Current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch to the next player
        self.current_player = 3 - self.current_player

        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Generate a visual representation of the game state
        available_numbers = [
            str(i + 1) for i in range(20) if self.available_numbers[i] == 1
        ]
        state_str = f"Available Numbers: {', '.join(available_numbers)}\n"
        parity_str = (
            "None"
            if self.parity_requirement == -1
            else ("Even" if self.parity_requirement == 0 else "Odd")
        )
        state_str += f"Parity Requirement: {parity_str}\n"
        state_str += f"Current Player: Player {self.current_player}\n"
        return state_str

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for i in range(20):
            if self.available_numbers[i] == 1:
                number = i + 1
                if self.parity_requirement == -1:
                    valid_actions.append(i)
                elif self.parity_requirement == 0 and number % 2 == 0:
                    valid_actions.append(i)
                elif self.parity_requirement == 1 and number % 2 == 1:
                    valid_actions.append(i)
        return valid_actions

    def _check_opponent_moves(self):
        # Check if the opponent can make a valid move
        for i in range(20):
            if self.available_numbers[i] == 1:
                number = i + 1
                if self.parity_requirement == -1:
                    return True
                elif self.parity_requirement == 0 and number % 2 == 0:
                    return True
                elif self.parity_requirement == 1 and number % 2 == 1:
                    return True
        return False

    def _get_observation(self):
        # Construct the observation array
        obs = np.append(self.available_numbers, self.parity_requirement).astype(
            np.int32
        )
        return obs
