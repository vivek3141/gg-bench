import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the action space: 6 possible actions
        # 0: Add 1, 1: Add 2, 2: Add 3, 3: Multiply by 1, 4: Multiply by 2, 5: Multiply by 3
        self.action_space = spaces.Discrete(6)

        # Define the observation space: current shared value
        self.observation_space = spaces.Box(
            low=0, high=21, shape=(1,), dtype=np.float32
        )

        # Action map to translate action indices to operations
        self.action_map = {
            0: ("add", 1),
            1: ("add", 2),
            2: ("add", 3),
            3: ("multiply", 1),
            4: ("multiply", 2),
            5: ("multiply", 3),
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_value = 0
        self.current_player = 1  # 1 or -1 to represent players
        self.done = False
        return (
            np.array([self.current_value], dtype=np.float32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            return (
                np.array([self.current_value], dtype=np.float32),
                0,
                True,
                False,
                {},
            )

        op, num = self.action_map[action]

        # Perform the operation
        if op == "add":
            new_value = self.current_value + num
        elif op == "multiply":
            new_value = self.current_value * num

        # Check for win/loss conditions
        if new_value == 21:
            self.current_value = new_value
            self.done = True
            return (
                np.array([self.current_value], dtype=np.float32),
                1,
                True,
                False,
                {},
            )
        elif new_value > 21:
            self.current_value = new_value
            self.done = True
            return (
                np.array([self.current_value], dtype=np.float32),
                -1,
                True,
                False,
                {},
            )
        else:
            # Valid move, update the current value and switch player
            self.current_value = new_value
            self.current_player *= -1
            return (
                np.array([self.current_value], dtype=np.float32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        return f"Current Value: {self.current_value}"

    def valid_moves(self):
        valid_actions = []
        for action in range(6):
            op, num = self.action_map[action]
            # Simulate the operation
            if op == "add":
                new_value = self.current_value + num
            elif op == "multiply":
                new_value = self.current_value * num
            # Check if the new value does not exceed 21
            if new_value <= 21:
                valid_actions.append(action)
        return valid_actions
