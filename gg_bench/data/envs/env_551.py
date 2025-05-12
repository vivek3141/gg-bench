import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Game settings
        self.start_number = 1
        self.target_number = 31
        self.allowed_operations = ["+1", "+2", "+3", "x2"]
        self.operations = [self.add1, self.add2, self.add3, self.mul2]

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.allowed_operations))
        self.observation_space = spaces.Box(
            low=np.array([self.start_number]),
            high=np.array([self.target_number]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.start_number
        self.current_player = 1  # Could be 1 or 2
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        # Apply the action
        operation = self.operations[action]
        new_number = operation(self.current_number)

        # Update the current number
        self.current_number = new_number

        # Check for loss
        if self.current_number > self.target_number:
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        # Check for win
        elif self.current_number == self.target_number:
            self.done = True
            reward = 1
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Game continues
            # Switch player
            self.current_player = 2 if self.current_player == 1 else 1
            reward = 0
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        return f"Current Number: {self.current_number}, Player {self.current_player}'s turn"

    def valid_moves(self):
        valid_actions = []
        for idx, operation in enumerate(self.operations):
            result = operation(self.current_number)
            if result <= self.target_number:
                valid_actions.append(idx)
        return valid_actions

    # Define the operations
    def add1(self, n):
        return n + 1

    def add2(self, n):
        return n + 2

    def add3(self, n):
        return n + 3

    def mul2(self, n):
        return n * 2
