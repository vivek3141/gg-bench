import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to choosing a multiplier from 2 to 9 inclusive
        self.action_space = spaces.Discrete(8)
        # Observation is the current running total
        self.observation_space = spaces.Box(low=1, high=1e9, shape=(1,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.running_total = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.running_total], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            return (
                np.array([self.running_total], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Check if the previous player has lost
        if self.running_total >= 1000:
            self.done = True
            reward = 1  # Current player wins
            return (
                np.array([self.running_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Map action to multiplier (2 to 9)
        multiplier = action + 2

        # Multiply the running total
        self.running_total *= multiplier

        # Switch to the next player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        # No need to check for loss here; it will be checked at the start of the next step
        return (
            np.array([self.running_total], dtype=np.int32),
            -10,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        return f"Current Total: {self.running_total}"

    def valid_moves(self):
        return list(range(8))  # All multipliers from 2 to 9 are always valid
