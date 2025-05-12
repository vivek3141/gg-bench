import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: numbers from 1 to 9, represented as indices from 0 to 8
        self.action_space = spaces.Discrete(9)

        # Observation space: array of size 18
        # First 9 entries: tower (numbers from 1 to 9 or 0 for empty)
        # Next 9 entries: number pool availability (1 for available, 0 for used)
        self.observation_space = spaces.Box(low=0, high=9, shape=(18,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the number pool: all numbers from 1 to 9 are available
        self.number_pool = np.ones(
            9, dtype=np.int32
        )  # indices 0 to 8 represent numbers 1 to 9

        # Initialize the tower: empty, represented by zeros
        self.tower = np.zeros(9, dtype=np.int32)  # maximum tower height is 9

        self.done = False

        self.tower_height = 0

        observation = self._get_observation()

        return observation, {}  # Returning observation and info dict

    def _get_observation(self):
        # Build the observation from tower and number_pool
        observation = np.concatenate((self.tower, self.number_pool))
        return observation

    def step(self, action):
        # Map action index to number
        number = action + 1  # action 0 corresponds to number 1, etc.

        if self.done:
            # If game is already over, return current observation
            return self._get_observation(), -10, True, False, {}

        # Check if the action is valid
        if self.number_pool[action] == 0:
            # Number has already been used
            self.done = True
            return self._get_observation(), -10, True, False, {}

        if not self._is_valid_move(number):
            # Move is invalid according to the stacking rules
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Move is valid, play it
        self.tower[self.tower_height] = number
        self.tower_height += 1

        # Remove the number from the number pool
        self.number_pool[action] = 0

        # Check if the game has ended (if next player has no valid moves)
        if self._check_game_end():
            # Current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        else:
            # Game continues
            return self._get_observation(), 0, False, False, {}

    def _is_valid_move(self, number):
        if self.tower_height == 0:
            # Any number can be placed if the tower is empty
            return True

        below_number = self.tower[self.tower_height - 1]

        # Stacking rules:
        # number can be placed if it is a divisor or multiple of below_number

        if below_number % number == 0 or number % below_number == 0:
            return True
        else:
            return False

    def _check_game_end(self):
        # The game ends if there are no valid moves left in the number pool
        for idx in range(9):
            if self.number_pool[idx] == 1:
                number = idx + 1
                if self._is_valid_move(number):
                    return False  # There is at least one valid move
        # No valid moves available
        return True

    def valid_moves(self):
        # Return a list of valid action indices
        valid_moves = []

        for idx in range(9):
            if self.number_pool[idx] == 1:
                number = idx + 1
                if self._is_valid_move(number):
                    valid_moves.append(idx)

        return valid_moves

    def render(self):
        # Return a visual representation of the environment state as a string
        tower_str = "Tower (bottom to top):\n"
        if self.tower_height == 0:
            tower_str += "Empty\n"
        else:
            for i in range(self.tower_height):
                number = self.tower[i]
                tower_str += f"Level {i+1}: {number}\n"

        pool_str = "Number Pool:\n"
        available_numbers = [
            str(idx + 1) for idx in range(9) if self.number_pool[idx] == 1
        ]
        pool_str += ", ".join(available_numbers)
        if not available_numbers:
            pool_str += "Empty"

        return tower_str + "\n" + pool_str
