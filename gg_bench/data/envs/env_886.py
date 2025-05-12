import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space represents multipliers from 2 to 9 (actions 0 to 7)
        self.action_space = spaces.Discrete(
            8
        )  # Actions 0-7 correspond to multipliers 2-9

        # The observation space is the current running total
        # We set the high value to a safe upper bound
        self.observation_space = spaces.Box(
            low=1, high=1e4, shape=(1,), dtype=np.float32
        )

        # Initialize the game state
        self.total = None
        self.current_player = None
        self.done = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total = 1.0
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.total], dtype=np.float32), {}  # Observation and info

    def step(self, action):
        if self.done:
            # If the game is over, return current state with zero reward
            return np.array([self.total], dtype=np.float32), 0, True, False, {}

        # Map action to multiplier
        multiplier = action + 2  # Map action 0-7 to multiplier 2-9

        # Update total
        self.total *= multiplier

        # Check for win condition
        if self.total >= 100:
            # Current player wins
            reward = 1
            self.done = True
            terminated = True
        else:
            # Valid move but game continues
            reward = -10  # Penalty for a valid move to encourage quicker wins
            terminated = False
            # Switch current player
            self.current_player = 2 if self.current_player == 1 else 1

        return np.array([self.total], dtype=np.float32), reward, terminated, False, {}

    def render(self):
        return f"Running Total is now {self.total}."

    def valid_moves(self):
        # Valid actions are always 0-7, corresponding to multipliers 2-9
        return list(range(8))
