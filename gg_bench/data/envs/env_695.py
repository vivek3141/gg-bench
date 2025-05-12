import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: integers from 0 to 7, corresponding to divisors 2 to 9
        self.action_space = spaces.Discrete(8)

        # Observation space: shared number and current player (0 or 1)
        self.observation_space = spaces.Dict(
            {
                "shared_number": spaces.Box(low=1, high=1e6, shape=(), dtype=np.int64),
                "current_player": spaces.Discrete(2),  # 0 or 1
            }
        )

        self.shared_number = None
        self.current_player = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = 100  # Starting number can be adjusted if needed
        self.current_player = 0  # Player 0 starts
        self.done = False
        observation = {
            "shared_number": self.shared_number,
            "current_player": self.current_player,
        }
        return observation, {}

    def step(self, action):
        if self.done:
            observation = {
                "shared_number": self.shared_number,
                "current_player": self.current_player,
            }
            return observation, 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            observation = {
                "shared_number": self.shared_number,
                "current_player": self.current_player,
            }
            return observation, reward, True, False, {}

        divisor = action + 2  # Map action index to divisor (2 to 9)

        # Perform the division and update the shared number
        self.shared_number = self.shared_number // divisor

        if self.shared_number == 1:
            # Current player wins
            self.done = True
            reward = 1
        else:
            reward = 0

        # Swap current player
        self.current_player = 1 - self.current_player

        observation = {
            "shared_number": self.shared_number,
            "current_player": self.current_player,
        }
        return observation, reward, self.done, False, {}

    def render(self):
        return f"Current number: {self.shared_number}, Player {self.current_player}'s turn."

    def valid_moves(self):
        # Return a list of valid action indices based on the current shared number
        valid_actions = []
        for action in range(8):  # Actions 0 to 7
            divisor = action + 2  # Divisors 2 to 9
            if self.shared_number // divisor >= 1:
                valid_actions.append(action)
        return valid_actions
