import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        # Actions: 0 for '+1', 1 for '*2'
        self.action_space = spaces.Discrete(2)

        # Observation is the current number, an integer scalar from 1 to 20
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([20]), dtype=np.int32
        )

        # Initialize the state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1  # Starting number
        self.done = False  # Game is not over
        self.current_player = 1  # Player 1 starts
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            # Game is over. Cannot take any actions.
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        # Apply the action
        if action == 0:  # '+1' action
            new_number = self.current_number + 1
        elif action == 1:  # '*2' action
            new_number = self.current_number * 2
        else:
            # Invalid action - Should not happen with Discrete(2) action space
            raise ValueError(f"Invalid action {action}. Action must be 0 or 1.")

        # Update the current number
        self.current_number = new_number

        # Check for game over conditions
        if self.current_number == 20:
            # Current player wins
            self.done = True
            reward = 1  # As per requirements
        elif self.current_number > 20:
            # Current player loses
            self.done = True
            reward = -10  # As per requirements
        else:
            # Game continues
            # Switch players
            self.current_player = 2 if self.current_player == 1 else 1
            reward = 0  # No reward
        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        return f"Current Number: {self.current_number}, Player {self.current_player}'s turn."

    def valid_moves(self):
        valid_actions = []
        # Check '+1' action (action = 0)
        if self.current_number + 1 <= 20:
            valid_actions.append(0)
        # Check '*2' action (action = 1)
        if self.current_number * 2 <= 20:
            valid_actions.append(1)
        return valid_actions
