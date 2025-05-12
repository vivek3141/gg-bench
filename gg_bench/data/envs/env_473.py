import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N):
        super(CustomEnv, self).__init__()

        self.N = N  # Target Number

        # Define action and observation space
        self.action_space = spaces.Discrete(
            8
        )  # Actions 0-7 correspond to multipliers 2-9

        # Observation space: [Current Number, Current Player]
        self.observation_space = spaces.Box(
            low=np.array([1, -1], dtype=np.int64),
            high=np.array([np.iinfo(np.int64).max, 1], dtype=np.int64),
            dtype=np.int64,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_number, self.current_player], dtype=np.int64), {}

    def step(self, action):
        if action not in range(8):
            # Invalid action
            self.done = True
            reward = -10
            return (
                np.array([self.current_number, self.current_player], dtype=np.int64),
                reward,
                True,
                False,
                {},
            )

        multiplier = action + 2  # Map action 0-7 to multiplier 2-9
        self.current_number *= multiplier

        if self.current_number >= self.N:
            # Current player wins
            reward = 1
            self.done = True
        else:
            # Game continues
            reward = 0
            self.current_player *= -1  # Switch player

        return (
            np.array([self.current_number, self.current_player], dtype=np.int64),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        return f"Current Number: {self.current_number}, Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        return list(range(8))  # Actions 0-7 are valid (multipliers 2 to 9)
