import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_number=20):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Add 1, 1 - Add 2, 2 - Add 3, 3 - Add 4,
        # 4 - Add 5, 5 - Multiply by 2, 6 - Multiply by 3
        self.action_space = spaces.Discrete(7)

        # Observation space: [current_number, target_number]
        self.observation_space = spaces.Box(
            low=np.array([1]),
            high=np.array([target_number]),
            shape=(1,),
            dtype=np.int32,
        )

        self.target_number = target_number
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.done = False
        # Return observation and info
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Apply the action
        if action == 0:
            # Add 1
            self.current_number += 1
        elif action == 1:
            # Add 2
            self.current_number += 2
        elif action == 2:
            # Add 3
            self.current_number += 3
        elif action == 3:
            # Add 4
            self.current_number += 4
        elif action == 4:
            # Add 5
            self.current_number += 5
        elif action == 5:
            # Multiply by 2
            self.current_number *= 2
        elif action == 6:
            # Multiply by 3
            self.current_number *= 3

        # Check for win condition
        if self.current_number == self.target_number:
            # Current player wins
            self.done = True
            return (np.array([self.current_number], dtype=np.int32), 1, True, False, {})

        # Check if current number exceeds target (should not happen)
        if self.current_number > self.target_number:
            # Should not happen with valid actions, but included for safety
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Continue game
        return (np.array([self.current_number], dtype=np.int32), 0, False, False, {})

    def render(self):
        return f"Current Number: {self.current_number}, Target Number: {self.target_number}"

    def valid_moves(self):
        valid_actions = []
        # Check addition actions
        for add_value, action in zip(range(1, 6), range(5)):
            if self.current_number + add_value <= self.target_number:
                valid_actions.append(action)
        # Check multiplication actions
        for multiplier, action in zip([2, 3], [5, 6]):
            if self.current_number * multiplier <= self.target_number:
                valid_actions.append(action)
        return valid_actions
