import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_divisor=3, max_length=10):
        super(CustomEnv, self).__init__()

        self.target_divisor = target_divisor
        self.max_length = max_length

        # Define action and observation space
        self.action_space = spaces.Discrete(10)  # Digits 0 through 9

        # Observation space includes the digits added so far (padded with -1) and the current modulo
        # The digits array has a fixed size of max_length
        # The last element is the current modulo with respect to the target divisor
        self.observation_space = spaces.Box(
            low=-1,
            high=9,
            shape=(self.max_length + 1,),
            dtype=np.int8,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.digits = []  # List to store the digits added so far
        self.current_modulo = (
            0  # Current modulo of the shared number with respect to the target divisor
        )
        self.length = 0  # Number of digits added so far
        self.done = False  # Flag to indicate if the game is over
        return self._get_observation(), {}  # Observation and info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        if self.done:
            return self._get_observation(), -10, True, False, {}

        # Append the action (digit) to the digits list
        self.digits.append(action)
        self.length += 1

        # Update the current modulo
        self.current_modulo = (self.current_modulo * 10 + action) % self.target_divisor

        # Check if the current player loses
        if self.current_modulo == 0:
            self.done = True
            return self._get_observation(), -10, True, False, {}
        elif self.length >= self.max_length:
            # Maximum length reached; current player loses
            self.done = True
            return self._get_observation(), -10, True, False, {}
        else:
            # Valid move; game continues
            return self._get_observation(), 0, False, False, {}

    def render(self):
        # Return a string representation of the current shared number
        if not self.digits:
            return "Shared Number: (empty)"
        else:
            shared_number = "".join(map(str, self.digits))
            return f"Shared Number: {shared_number}"

    def valid_moves(self):
        # All digits from 0 to 9 are valid moves
        return list(range(10))

    def _get_observation(self):
        # Create an observation array with digits and the current modulo
        obs = np.full(self.max_length + 1, -1, dtype=np.int8)
        obs[: len(self.digits)] = self.digits
        obs[-1] = self.current_modulo
        return obs
