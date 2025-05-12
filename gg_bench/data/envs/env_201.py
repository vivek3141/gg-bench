import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_divisor=3, max_length=20):
        super(CustomEnv, self).__init__()

        self.target_divisor = target_divisor
        self.max_length = max_length

        # Define action space: 0 for '0', 1 for '1'
        self.action_space = spaces.Discrete(2)

        # Define observation space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.max_length,), dtype=np.int8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.binary_string = []
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Return the initial observation and info
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            # If game is over, return with a penalty
            return self._get_observation(), -10, True, False, {}

        if action not in [0, 1]:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Enforce the rule: the binary string must not start with '0' unless it is '0' itself
        if len(self.binary_string) == 0 and action == 0:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Append the action as the new bit
        self.binary_string.append(action)

        # Convert binary string to decimal
        binary_str = "".join(map(str, self.binary_string))
        decimal_value = int(binary_str, 2)

        # Check for victory conditions
        if len(self.binary_string) >= 4 and decimal_value % self.target_divisor == 0:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}
        elif len(self.binary_string) >= self.max_length:
            # Reached maximum length without a winner
            self.done = True
            reward = 0
            return self._get_observation(), reward, True, False, {}
        else:
            # Switch players
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
            reward = 0
            return self._get_observation(), reward, False, False, {}

    def render(self):
        binary_str = "".join(map(str, self.binary_string))
        decimal_value = int(binary_str, 2) if binary_str else 0
        return f"Current Binary String: {binary_str} (Decimal: {decimal_value})"

    def valid_moves(self):
        if self.done:
            return []
        valid_actions = [0, 1]
        if len(self.binary_string) == 0:
            # First move cannot be '0' unless it's the only bit
            valid_actions = [1]
        return valid_actions

    def _get_observation(self):
        # Pad the binary string to the maximum length
        obs = np.zeros(self.max_length, dtype=np.int8)
        obs[: len(self.binary_string)] = self.binary_string
        return obs
