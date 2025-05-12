import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - increment hundreds, 1 - increment tens, 2 - increment units
        self.action_space = spaces.Discrete(3)

        # Observation space: three digits from 0 to 9
        self.observation_space = spaces.Box(low=0, high=9, shape=(3,), dtype=np.int32)

        # Maximum number of steps to prevent infinite games
        self.max_steps = 100

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.safe_code = np.array([0, 0, 0], dtype=np.int32)
        self.current_player = 1  # 1 or -1 to represent two players
        self.terminated = False
        self.truncated = False
        self.step_count = 0
        return self.safe_code, {}  # Observation and info

    def step(self, action):
        if self.terminated or self.truncated:
            # If the game is over, no more actions can be taken
            return self.safe_code, 0, self.terminated, self.truncated, {}

        # Increment step count
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.truncated = True
            return self.safe_code, 0, False, True, {}

        if action not in [0, 1, 2]:
            # Invalid action
            self.terminated = True
            return self.safe_code, -10, True, False, {}

        # Increment the chosen digit
        self.safe_code[action] = (self.safe_code[action] + 1) % 10

        # Check for win condition
        if np.all(self.safe_code == 7):
            self.terminated = True
            return self.safe_code, 1, True, False, {}

        # Switch to the other player
        self.current_player *= -1

        return self.safe_code, 0, False, False, {}

    def render(self):
        # Returns a string representation of the current safe code
        code_str = f"Current Safe Code: {self.safe_code[0]} {self.safe_code[1]} {self.safe_code[2]}"
        return code_str

    def valid_moves(self):
        # All digits can always be incremented
        return [0, 1, 2]
