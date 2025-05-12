import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, max_length=9):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are digits 1-9, represented as action indices 0-8
        self.action_space = spaces.Discrete(9)

        # Observation space: vector of digits, padded to max_length
        self.max_length = max_length
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.max_length,), dtype=np.int8
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N_digits = []
        self.current_player = 1  # Player 1 starts
        self.done = False
        # Build observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if not self.action_space.contains(action) or self.done:
            # Invalid action or episode is over
            return self._get_observation(), -10.0, True, False, {}

        digit = action + 1  # Map action indices 0-8 to digits 1-9
        self.N_digits.append(digit)

        # Check if N is a multi-digit palindrome
        if len(self.N_digits) >= 2 and self.N_digits == self.N_digits[::-1]:
            # Current player wins
            self.done = True
            reward = 1.0
        elif len(self.N_digits) >= self.max_length:
            # Reached maximum length without forming a palindrome
            self.done = True
            reward = -10.0
        else:
            # Game continues
            reward = 0.0
            # Switch player
            self.current_player *= -1

        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def render(self):
        if not self.N_digits:
            N_str = "0"
        else:
            N_str = "".join(str(d) for d in self.N_digits)
        return f"Current number N: {N_str}"

    def valid_moves(self):
        if self.done:
            return []
        else:
            return list(range(9))  # Actions 0-8 correspond to digits 1-9

    def _get_observation(self):
        padded_N = self.N_digits + [0] * (self.max_length - len(self.N_digits))
        return np.array(padded_N, dtype=np.int8)
