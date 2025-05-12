import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Actions: 0 - Add 1, 1 - Multiply by 2
        self.action_space = spaces.Discrete(2)

        # Observation: the cumulative total, an integer between 1 and 100
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([100]), dtype=np.int32
        )

        # Initialize the total and other variables
        self.total = None
        self.current_player = None
        self.done = None

        # Reset the environment to its initial state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.total]), {}  # Observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, return the current state
            return np.array([self.total]), 0, True, False, {}

        # Apply the action
        if action == 0:
            # Add 1
            new_total = self.total + 1
        elif action == 1:
            # Multiply by 2
            new_total = self.total * 2
        else:
            # Invalid action (should not occur with Discrete(2))
            raise ValueError("Invalid action.")

        # Check for invalid move (exceeds 100)
        if new_total > 100:
            self.done = True
            reward = -10  # Player loses
            return np.array([self.total]), reward, True, False, {}

        # Update the total
        self.total = new_total

        # Check for winning move
        if self.total == 100:
            self.done = True
            reward = 1  # Player wins
            return np.array([self.total]), reward, True, False, {}

        # Valid move, game continues
        reward = 0

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        return np.array([self.total]), reward, False, False, {}

    def render(self):
        # Return a string representation of the current state
        return (
            f"Current total is {self.total}. "
            f"It's Player {self.current_player}'s turn."
        )

    def valid_moves(self):
        # Return a list of valid actions
        valid_actions = []
        if self.total + 1 <= 100:
            valid_actions.append(0)  # Add 1 is valid
        if self.total * 2 <= 100:
            valid_actions.append(1)  # Multiply by 2 is valid
        return valid_actions
