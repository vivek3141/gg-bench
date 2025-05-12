import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, goal_number=20):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 0 => Add 1, 1 => Multiply by 2
        self.action_space = spaces.Discrete(2)

        # Observation space: current number, bounded between 1 and goal_number * 2
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([goal_number * 2]), dtype=np.int32
        )

        self.goal_number = goal_number

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}  # Observation, info

    def step(self, action):
        if self.done:
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Validate action
        if action not in [0, 1]:
            reward = -10
            self.done = True
            terminated = True
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                terminated,
                False,
                {},
            )

        # Apply the selected operation
        if action == 0:
            # Add 1
            new_number = self.current_number + 1
        elif action == 1:
            # Multiply by 2
            new_number = self.current_number * 2

        # Check for win or loss
        if new_number == self.goal_number:
            reward = 1  # Current player wins
            self.done = True
            terminated = True
        elif new_number > self.goal_number:
            reward = -10  # Current player loses
            self.done = True
            terminated = True
        else:
            reward = 0
            terminated = False

        # Update the game state
        self.current_number = new_number

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            terminated,
            False,
            {},
        )

    def render(self):
        return f"Player {self.current_player}'s turn. Current number: {self.current_number}"

    def valid_moves(self):
        return [0, 1]  # Both operations are always available
