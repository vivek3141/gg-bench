import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=60, max_factor=100):
        super(CustomEnv, self).__init__()

        # Initialize starting number and maximum factor
        self.starting_number = starting_number
        self.max_factor = max_factor  # Maximum factor to consider for action mapping

        # Define action space: actions correspond to selecting factors from 2 to max_factor
        self.action_space = spaces.Discrete(
            self.max_factor - 1
        )  # Excludes 1 and max_factor + 1

        # Define observation space: [Current Number, Current Player, Used Factors]
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([np.inf] * (2 + self.max_factor - 1)),
            dtype=np.float32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Allow setting of starting number via options
        if options is not None and "starting_number" in options:
            self.starting_number = options["starting_number"]

        self.current_number = self.starting_number
        self.used_factors = np.zeros(
            self.max_factor - 1, dtype=np.int32
        )  # Index corresponds to factor - 2
        self.current_player = 1  # 1 or 2
        self.done = False

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        factor = action + 2  # Map action index to factor (since factors start from 2)

        # Check for valid action
        if self.done or not self._is_valid_factor(factor):
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Apply action: divide the Current Number by the selected factor
        self.current_number = self.current_number // factor
        self.used_factors[factor - 2] = 1  # Mark factor as used

        # Check if opponent has valid moves
        if not self._has_valid_moves():
            # Current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Continue game by switching players
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        # Return a string representation of the game state
        current_state = (
            f"Current Number: {self.current_number}\n"
            f"Used Factors: {self._get_used_factors_list()}\n"
            f"Current Player: Player {self.current_player}"
        )
        return current_state

    def valid_moves(self):
        # Return a list of valid action indices
        valid_factors = self._get_valid_factors()
        return [factor - 2 for factor in valid_factors]  # Map factors to action indices

    # Helper methods
    def _get_observation(self):
        # Construct the observation array
        obs = np.zeros(2 + self.max_factor - 1, dtype=np.float32)
        obs[0] = self.current_number
        obs[1] = self.current_player
        obs[2:] = self.used_factors
        return obs

    def _is_valid_factor(self, factor):
        # Check if the factor is a valid move
        if factor <= 1 or factor >= self.current_number:
            return False
        if self.current_number % factor != 0:
            return False
        if self.used_factors[factor - 2] == 1:
            return False
        return True

    def _get_valid_factors(self):
        # Get a list of valid factors for the current number
        factors = []
        for i in range(2, min(self.current_number, self.max_factor + 1)):
            if self.current_number % i == 0 and self.used_factors[i - 2] == 0:
                factors.append(i)
        return factors

    def _has_valid_moves(self):
        # Check if there are any valid moves left for the current player
        return len(self._get_valid_factors()) > 0

    def _get_used_factors_list(self):
        # Get a list of factors that have been used
        return [i + 2 for i, used in enumerate(self.used_factors) if used == 1]
