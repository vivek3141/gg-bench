import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):

    def __init__(self, target_number=100):
        super(CustomEnv, self).__init__()

        self.target_number = target_number

        # Define action space: multipliers from 2 to 9 inclusive, mapped to actions 0 to 7
        self.action_space = spaces.Discrete(8)

        # Observation space includes current number and current player (1 or 2)
        self.observation_space = spaces.Box(
            low=np.array([1, 1]),
            high=np.array([np.iinfo(np.int64).max, 2]),
            dtype=np.int64,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.done = False
        self.current_player = 1  # Player 1 starts
        return np.array([self.current_number, self.current_player], dtype=np.int64), {}

    def step(self, action):
        if self.done:
            # Trying to step after the game is over
            return (
                np.array([self.current_number, self.current_player], dtype=np.int64),
                -10,
                True,
                False,
                {},
            )
        if action not in range(8):
            # Invalid action
            self.done = True
            return (
                np.array([self.current_number, self.current_player], dtype=np.int64),
                -10,
                True,
                False,
                {},
            )

        # Apply action
        multiplier = action + 2  # Actions 0-7 map to multipliers 2-9
        self.current_number *= multiplier

        # Check for victory
        if self.current_number >= self.target_number:
            self.done = True
            reward = 1  # Current player wins
            return (
                np.array([self.current_number, self.current_player], dtype=np.int64),
                reward,
                True,
                False,
                {},
            )
        else:
            # Switch player
            self.current_player = 2 if self.current_player == 1 else 1
            reward = -1  # Penalize per move to encourage faster wins
            return (
                np.array([self.current_number, self.current_player], dtype=np.int64),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        return f"Current Number: {self.current_number}, Current Player: Player {self.current_player}"

    def valid_moves(self):
        if self.done:
            return []
        else:
            return list(range(8))  # Actions 0-7 are valid moves
