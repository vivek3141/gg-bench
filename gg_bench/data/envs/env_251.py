import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the target total
        self.target_total = 23

        # Define action and observation spaces
        self.action_space = spaces.Discrete(9)  # Actions 0-8 represent numbers 1-9

        # Observation space: [running_total, current_player]
        self.observation_space = spaces.Box(
            low=np.array([0, -1]), high=np.array([self.target_total, 1]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.running_total = 0
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return np.array([self.running_total, self.current_player], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.running_total, self.current_player], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Convert action (0-8) to number (1-9)
        number = action + 1

        # Validate move
        if number < 1 or number > 9:
            self.done = True
            reward = -10
            return (
                np.array([self.running_total, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        new_total = self.running_total + number

        if new_total > self.target_total:
            self.done = True
            reward = -10  # Player loses
            return (
                np.array([self.running_total, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        self.running_total = new_total

        if self.running_total == self.target_total:
            self.done = True
            reward = 1  # Player wins
            return (
                np.array([self.running_total, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # No win or lose, game continues
        reward = 0

        # Switch to the other player
        self.current_player *= -1

        return (
            np.array([self.running_total, self.current_player], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        return f"Running Total: {self.running_total}, Player: {'1' if self.current_player == 1 else '2'}"

    def valid_moves(self):
        # Return list of valid moves (action indices) that do not exceed the target total
        return [
            i for i in range(9) if self.running_total + (i + 1) <= self.target_total
        ]
