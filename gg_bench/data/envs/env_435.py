import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_divisor=7):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: digits 1-9 (indices 0-8)
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # First 9 entries: current number (digits placed so far, padded with zeros)
        # Next 9 entries: digit pool (1 if available, 0 if used)
        self.observation_space = spaces.Box(low=0, high=9, shape=(18,), dtype=np.int32)

        # Set target divisor
        self.target_divisor = target_divisor

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.digit_pool = np.ones(9, dtype=np.int32)  # 1 means available, 0 means used
        self.current_number = []
        self.current_player = 1
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}
        digit = action + 1  # Map action index to digit (1-9)
        if self.digit_pool[action] == 0:
            # Invalid move: digit already used
            self.done = True
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move
        self.current_number.append(digit)
        self.digit_pool[action] = 0  # Mark digit as used

        # Form the current number
        number = int("".join(map(str, self.current_number)))

        # Check for victory condition
        if number % self.target_divisor == 0:
            self.done = True
            return self._get_obs(), 1, True, False, {}  # Current player wins

        # Check for forced loss (all digits used)
        if np.all(self.digit_pool == 0):
            # All digits used, no victory
            self.done = True
            return self._get_obs(), -1, True, False, {}  # Current player loses

        # Game continues
        self.current_player *= -1  # Switch player
        return (
            self._get_obs(),
            0,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        number_str = "".join(map(str, self.current_number))
        digit_pool_str = " ".join(
            [str(i + 1) for i in range(9) if self.digit_pool[i] == 1]
        )
        return f"Current Number: {number_str}\nAvailable Digits: {digit_pool_str}\n"

    def valid_moves(self):
        return [i for i in range(9) if self.digit_pool[i] == 1]

    def _get_obs(self):
        # Current number padded to length 9
        current_number_padded = np.pad(
            self.current_number, (0, 9 - len(self.current_number)), "constant"
        )
        observation = np.concatenate([current_number_padded, self.digit_pool])
        return observation
