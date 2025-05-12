import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: numbers from 2 to 100, represented as actions 0 to 98
        self.action_space = spaces.Discrete(99)

        # Observation space:
        # Elements 0 to 98: binary indicators for numbers 2 to 100 (0: not played, 1: played)
        # Element 99: normalized last number played (0 if no last number)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(100,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.played_numbers = np.zeros(99, dtype=np.float32)  # Numbers 2 to 100
        self.shared_list = []
        self.current_player = 1  # 1 or -1
        self.last_number = 0  # 0 indicates no number yet
        self.done = False

        # Observation: concatenate played_numbers and normalized last_number
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Convert action to number
        number = action + 2  # Actions 0-98 correspond to numbers 2-100

        # Check if number has already been played
        if self.played_numbers[action] == 1:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if action is valid
        if not self._is_valid_move(number):
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move
        # Update state
        self.played_numbers[action] = 1
        self.shared_list.append(number)
        self.last_number = number

        # Check if opponent has valid moves
        opponent_valid_moves = self._get_valid_moves()
        if len(opponent_valid_moves) == 0:
            # Opponent cannot move, current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}
        else:
            # Switch player
            self.current_player *= -1
            return self._get_observation(), 0, False, False, {}

    def render(self):
        shared_list_str = ", ".join(map(str, self.shared_list))
        return f"Shared List: [{shared_list_str}]"

    def valid_moves(self):
        return self._get_valid_moves()

    def _get_observation(self):
        # Normalize last_number to range [0, 1]
        normalized_last_number = (
            (self.last_number - 2) / 98 if self.last_number > 0 else 0.0
        )
        observation = np.concatenate([self.played_numbers, [normalized_last_number]])
        return observation

    def _is_valid_move(self, number):
        if self.last_number == 0:
            # First move, any number is valid
            return True
        else:
            # Number must be a factor or multiple of last_number
            return (self.last_number % number == 0) or (number % self.last_number == 0)

    def _get_valid_moves(self):
        valid_moves = []
        for action in range(99):
            if self.played_numbers[action] == 0:
                number = action + 2
                if self._is_valid_move(number):
                    valid_moves.append(action)
        return valid_moves
