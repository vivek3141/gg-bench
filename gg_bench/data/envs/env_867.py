import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, D=7, max_steps=20):
        super(CustomEnv, self).__init__()

        self.D = D  # Target Divisor
        self.max_steps = max_steps  # Maximum number of steps before game ends

        # Define action space: digits 0-9
        self.action_space = spaces.Discrete(10)

        # Define observation space: N_mod_D (0 to D-1)
        self.observation_space = spaces.Box(
            low=0, high=self.D - 1, shape=(1,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.N_mod_D = 0  # Current value of N mod D
        self.current_player = 1  # 1 or -1 to represent players
        self.done = False  # Game over flag
        self.steps = 0  # Step counter

        return np.array([self.N_mod_D]), {}  # Observation and info

    def step(self, action):
        # Check if game is already over
        if self.done:
            return np.array([self.N_mod_D]), -10, True, False, {}

        # Check if action is valid
        if action < 0 or action > 9:
            self.done = True
            return np.array([self.N_mod_D]), -10, True, False, {}

        self.steps += 1

        # Update N_mod_D based on appended digit
        self.N_mod_D = (self.N_mod_D * 10 + action) % self.D

        # Check if current player wins
        if self.N_mod_D == 0:
            self.done = True
            return np.array([self.N_mod_D]), 1, True, False, {}

        # Check if maximum steps reached (optional rule)
        if self.steps >= self.max_steps:
            self.done = True
            return np.array([self.N_mod_D]), 0, True, False, {}

        # Switch to the other player
        self.current_player *= -1

        return (
            np.array([self.N_mod_D]),
            0,
            False,
            False,
            {},
        )  # Observation, reward, done, info

    def render(self):
        return (
            f"Current N mod D: {self.N_mod_D} mod {self.D}, Steps taken: {self.steps}"
        )

    def valid_moves(self):
        if self.done:
            return []
        else:
            return list(range(10))  # Digits 0-9 are valid moves
