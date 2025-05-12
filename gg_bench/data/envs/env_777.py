import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to multipliers from 2 to 9 (action 0 -> multiplier 2, ..., action 7 -> multiplier 9)
        self.action_space = spaces.Discrete(8)

        # Observation space consists of [current_total, current_player]
        # current_total ranges from 1 to 1,000,000, current_player is 0 or 1
        self.observation_space = spaces.Box(
            low=np.array([1, 0]), high=np.array([1_000_000, 1]), dtype=np.float32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_total = 1
        self.current_player = 0  # Player 0 starts
        self.done = False
        return np.array([self.current_total, self.current_player], dtype=np.float32), {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return (
                np.array([self.current_total, self.current_player], dtype=np.float32),
                0,
                True,
                False,
                {},
            )

        # Validate action
        if action not in self.valid_moves():
            self.done = True
            return (
                np.array([self.current_total, self.current_player], dtype=np.float32),
                -10,
                True,
                False,
                {},
            )

        # Map action to multiplier (action 0 corresponds to multiplier 2)
        multiplier = action + 2
        self.current_total *= multiplier

        # Check for victory
        if self.current_total >= 100:
            self.done = True
            reward = 1
            return (
                np.array([self.current_total, self.current_player], dtype=np.float32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Switch player
            self.current_player = 1 - self.current_player
            reward = 0
            return (
                np.array([self.current_total, self.current_player], dtype=np.float32),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        return (
            f"Player {self.current_player}'s turn. Current Total: {self.current_total}"
        )

    def valid_moves(self):
        if self.done:
            return []
        else:
            return list(range(self.action_space.n))
