import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Add One, 1 - Double
        self.action_space = spaces.Discrete(2)

        # Define observation space: Current number between 1 and 20
        self.observation_space = spaces.Box(low=1, high=20, shape=(1,), dtype=np.int32)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts; can represent players as 1 and -1
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}  # Observation, info

    def step(self, action):
        # Check if action is valid
        if action not in [0, 1]:
            # Invalid action index
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move: Action would exceed 20
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Apply the action
        if action == 0:
            # Add One
            self.current_number += 1
        elif action == 1:
            # Double
            self.current_number *= 2

        # Check for win condition
        if self.current_number == 20:
            # Current player wins
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )
        elif self.current_number > 20:
            # Should not happen due to action validation, but handle just in case
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )
        else:
            # Switch player
            self.current_player *= -1
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        state_str = f"Current Number: {self.current_number}\n"
        state_str += f"Player {1 if self.current_player == 1 else 2}'s turn.\n"
        return state_str

    def valid_moves(self):
        valid_actions = []
        if self.current_number + 1 <= 20:
            valid_actions.append(0)  # Add One is valid
        if self.current_number * 2 <= 20:
            valid_actions.append(1)  # Double is valid
        return valid_actions
