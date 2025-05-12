import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: actions 0 to 7 correspond to multipliers 2 to 9
        self.action_space = spaces.Discrete(8)  # actions {0,1,...,7}

        # Observation space: the current total
        self.observation_space = spaces.Box(
            low=1, high=np.inf, shape=(1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.total], dtype=np.float32), {}  # Observation, info

    def step(self, action):
        # Map action to multiplier (0-7) -> (2-9)
        multiplier = action + 2

        # Update total
        self.total *= multiplier

        # Check if current player loses
        if self.total >= 1000:
            self.done = True
            reward = 1  # Current player wins (since opponent caused the loss by forcing the player into losing)
            return (
                np.array([self.total], dtype=np.float32),
                reward,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Game continues
        reward = -10  # The current player has played a valid move
        self.current_player = 2 if self.current_player == 1 else 1  # Switch player
        return (
            np.array([self.total], dtype=np.float32),
            reward,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def render(self):
        return (
            f"Current total: {self.total}, Current player: Player {self.current_player}"
        )

    def valid_moves(self):
        return list(range(8))  # Actions 0 to 7 correspond to multipliers 2-9
