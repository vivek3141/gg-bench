import numpy as np
import gymnasium as gym
from gymnasium import spaces


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to choosing numbers from 1 to 50 (indices 0 to 49)
        self.action_space = spaces.Discrete(50)

        # Observation space consists of the running total and availability of numbers 1-50
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(51,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.running_total = 0
        self.available_numbers = np.ones(
            50, dtype=np.int32
        )  # Numbers 1 to 50 are available
        self.done = False
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if action is valid
        if action < 0 or action >= 50:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        if self.available_numbers[action] == 0:
            # Number already used
            self.done = True
            return self._get_obs(), -10, True, False, {}

        number = action + 1  # Actual number chosen

        if self.running_total + number > 100:
            # Move exceeds running total limit
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid move, update the game state
        self.running_total += number
        self.available_numbers[action] = 0

        # Check for a win condition
        if is_prime(self.running_total):
            self.done = True
            return self._get_obs(), 1, True, False, {}  # Current player wins

        # Check if running total has reached or exceeded 100
        if self.running_total >= 100:
            self.done = True
            return self._get_obs(), 0, True, False, {}  # Current player loses

        # Check if there are any valid moves left for the next turn
        if len(self.valid_moves()) == 0:
            self.done = True
            return self._get_obs(), 0, True, False, {}  # Current player loses

        # Game continues without any reward
        return self._get_obs(), 0, False, False, {}

    def render(self):
        output = f"Running Total: {self.running_total}\n"
        output += "Available Numbers: "
        available_nums = [
            str(i + 1) for i in range(50) if self.available_numbers[i] == 1
        ]
        output += ", ".join(available_nums)
        return output

    def valid_moves(self):
        valid = []
        for action in range(50):
            if self.available_numbers[action] == 1:
                number = action + 1
                if self.running_total + number <= 100:
                    valid.append(action)
        return valid

    def _get_obs(self):
        observation = np.concatenate(([self.running_total], self.available_numbers))
        return observation
