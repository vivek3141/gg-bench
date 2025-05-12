import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            4
        )  # Actions 0-3 correspond to subtracting numbers 1-4
        self.observation_space = spaces.Box(low=0, high=23, shape=(1,), dtype=np.int32)

        # Allowed numbers to subtract
        self.allowed_numbers = [1, 2, 3, 4]

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total = 23  # Starting total
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.total], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        number_to_subtract = self.allowed_numbers[action]

        if self.done:
            # The game has already ended
            return np.array([self.total], dtype=np.int32), 0, True, False, {}

        if number_to_subtract > self.total:
            # Invalid move: cannot subtract more than the current total
            self.done = True
            return np.array([self.total], dtype=np.int32), -10, True, False, {}

        # Valid move
        self.total -= number_to_subtract

        if self.total == 0:
            # Current player wins
            self.done = True
            return np.array([self.total], dtype=np.int32), 1, True, False, {}

        else:
            # Game continues; switch to the other player
            self.current_player = 3 - self.current_player  # Switches between 1 and 2
            return np.array([self.total], dtype=np.int32), -10, False, False, {}

    def render(self):
        return (
            f"Current total: {self.total}, Current player: Player {self.current_player}"
        )

    def valid_moves(self):
        # Return a list of valid action indices based on the current total
        return [i for i, num in enumerate(self.allowed_numbers) if num <= self.total]
