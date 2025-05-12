import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete(20), actions from 0 to 19 correspond to numbers 1 to 20
        self.action_space = spaces.Discrete(20)

        # Observation space: Box space
        # - First 20 elements: availability of numbers 1 to 20 (1 for available, 0 for taken)
        # - Element 20: last opponent's number (integer from 0 to 20)
        # Note: high values for elements 0-19 is 1 (availability), for element 20 is 20
        low = np.zeros(21, dtype=np.int32)
        high = np.concatenate(
            (np.ones(20, dtype=np.int32), np.array([20], dtype=np.int32))
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the availability of numbers 1-20 to 1 (available)
        self.availability = np.ones(20, dtype=np.int32)

        # Last opponent's number, 0 indicates no number yet (for first move)
        self.last_opponent_number = 0

        # Current player: 1 or 2, Player 1 starts
        self.current_player = 1

        # Game over flag
        self.done = False

        # Build the observation
        observation = np.concatenate(
            (self.availability, np.array([self.last_opponent_number], dtype=np.int32))
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        # Convert action index to number (1-20)
        selected_number = action + 1

        # Check if game is already over
        if self.done:
            reward = 0
            return self._get_observation(), reward, self.done, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move, current player loses
            self.done = True
            reward = -10
            return self._get_observation(), reward, self.done, False, {}

        # Update the availability
        self.availability[action] = 0  # Number is now taken

        # Update the last opponent's number to be the number selected in this turn
        previous_opponent_number = (
            self.last_opponent_number
        )  # Keep for checking valid moves
        self.last_opponent_number = selected_number

        # Check if opponent has any valid moves
        opponent_valid_actions = self._get_valid_moves_for_number(selected_number)
        if len(opponent_valid_actions) == 0:
            # Opponent cannot move, current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, self.done, False, {}

        # Switch players
        self.current_player = 2 if self.current_player == 1 else 1

        # Game continues
        reward = 0
        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        output = "Available Numbers: "
        available_numbers = [str(i + 1) for i in range(20) if self.availability[i]]
        output += ", ".join(available_numbers) + "\n"
        output += f"Last Opponent's Number: {self.last_opponent_number}\n"
        output += f"Current Player: Player {self.current_player}\n"
        return output

    def valid_moves(self):
        # Return list of valid action indices (0-19)
        return self._get_valid_moves_for_number(self.last_opponent_number)

    def _get_valid_moves_for_number(self, opponent_number):
        valid_actions = []
        for i in range(20):
            if self.availability[i]:
                number = i + 1  # numbers 1 to 20
                if opponent_number == 0:
                    # First move, all numbers are valid
                    valid_actions.append(i)
                else:
                    # Check if number is neither a factor nor multiple of opponent's number
                    if not self._is_factor_or_multiple(number, opponent_number):
                        valid_actions.append(i)
        return valid_actions

    def _is_factor_or_multiple(self, number, opponent_number):
        if opponent_number == 0:
            return False  # First move, no restrictions
        if number == 1 and opponent_number == 1:
            return True  # Special case: 1 is a factor of 1
        if number == 1 and opponent_number != 1:
            return False  # 1 is valid unless opponent_number is 1
        if opponent_number == 1:
            if number == 1:
                return True
            else:
                return False
        if opponent_number % number == 0 or number % opponent_number == 0:
            return True
        else:
            return False

    def _get_observation(self):
        observation = np.concatenate(
            (self.availability, np.array([self.last_opponent_number], dtype=np.int32))
        )
        return observation
