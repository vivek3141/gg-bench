import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_number=7, max_digits=20, max_steps=100):
        super(CustomEnv, self).__init__()

        self.MAX_DIGITS = max_digits
        self.target_number = target_number
        self.max_steps = max_steps

        # Action space: 0-19
        # Actions 0-9: Add digit 0-9 to left
        # Actions 10-19: Add digit 0-9 to right
        self.action_space = spaces.Discrete(20)

        # Observation space: Array of digits, length MAX_DIGITS
        # Digits 0-9 or -1 for empty
        self.observation_space = spaces.Box(
            low=-1, high=9, shape=(self.MAX_DIGITS,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = []
        self.current_player = 1
        self.done = False
        self.truncated = False
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.full((self.MAX_DIGITS,), -1, dtype=np.int32)
        obs[: len(self.shared_number)] = self.shared_number
        return obs

    def step(self, action):
        if self.done or self.truncated:
            return self._get_obs(), 0, self.done, self.truncated, {}

        # Increment step count
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.truncated = True
            return self._get_obs(), -10, True, True, {}

        # Decode action
        if action < 0 or action >= 20:
            # Invalid action
            return self._get_obs(), -10, True, False, {}
        digit = action % 10
        position = "left" if action < 10 else "right"

        if len(self.shared_number) >= self.MAX_DIGITS:
            # Invalid move: number is maximum length
            return self._get_obs(), -10, True, False, {}

        # Apply action
        if position == "left":
            self.shared_number.insert(0, digit)
        else:
            self.shared_number.append(digit)

        # Construct number as string to preserve leading zeros
        num_str = "".join(map(str, self.shared_number))
        if len(num_str.strip("0")) == 0:
            num = 0
        else:
            num = int(num_str)

        # Check for victory
        if num % self.target_number == 0 and num != 0:
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}
        else:
            reward = -10
            self.current_player *= -1  # Switch player
            return self._get_obs(), reward, False, False, {}

    def render(self):
        num_str = "".join(map(str, self.shared_number))
        return f"Shared number is: '{num_str}'"

    def valid_moves(self):
        if len(self.shared_number) >= self.MAX_DIGITS:
            return []
        else:
            return list(range(20))
