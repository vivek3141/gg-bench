import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions 0 to 8 correspond to numbers 1 to 9
        self.action_space = spaces.Discrete(9)
        self.MAX_TOWER_HEIGHT = 100  # Maximum height of the tower
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.MAX_TOWER_HEIGHT,), dtype=np.int32
        )

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tower = []  # Start with an empty tower
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        number = action + 1  # Map action index to number 1-9

        # Check if the move is valid
        if not self._is_valid_move(number):
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_observation(), reward, True, False, {}

        # Place the number on top of the tower
        self.tower.append(number)

        # Check if the opponent has any valid moves
        if not self._opponent_has_valid_moves():
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}

        # Switch to the next player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        return self._get_observation(), 0, False, False, {}

    def render(self):
        tower_str = "Tower (from bottom to top): " + " -> ".join(map(str, self.tower))
        return tower_str

    def valid_moves(self):
        return [i for i in range(9) if self._is_valid_move(i + 1)]

    def _get_observation(self):
        obs = np.zeros(self.MAX_TOWER_HEIGHT, dtype=np.int32)
        obs[: len(self.tower)] = self.tower
        return obs

    def _is_valid_move(self, number):
        if not 1 <= number <= 9:
            return False  # Number must be between 1 and 9
        if not self.tower:
            return True  # Any number is valid if the tower is empty
        top_number = self.tower[-1]
        if number % top_number == 0 or top_number % number == 0:
            return True  # Valid if number is a factor or multiple of the top
        return False

    def _opponent_has_valid_moves(self):
        top_number = self.tower[-1]
        for number in range(1, 10):
            if number % top_number == 0 or top_number % number == 0:
                return True  # Opponent has at least one valid move
        return False  # Opponent cannot make a valid move
