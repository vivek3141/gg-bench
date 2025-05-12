import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are 0 to 3 representing choices of numbers 1 to 4
        self.action_space = spaces.Discrete(4)
        # Observation is the cumulative total, ranging from 0 to 23
        self.observation_space = spaces.Box(
            low=np.array([0]), high=np.array([23]), shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.cumulative_total = None
        self.current_player = None
        self.done = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the cumulative total and game status
        self.cumulative_total = 0
        self.current_player = 1  # Player 1 starts
        self.done = False

        return np.array([self.cumulative_total]), {}  # Observation and info

    def step(self, action):
        if self.done:
            # If the game is over, ignore further actions
            return np.array([self.cumulative_total]), 0, True, False, {}

        if action not in [0, 1, 2, 3]:
            # Invalid action (not in action space)
            self.done = True
            return np.array([self.cumulative_total]), -10, True, False, {}

        chosen_number = action + 1  # Map action to number between 1 and 4

        if action not in self.valid_moves():
            # Action would cause the total to exceed 23
            self.done = True
            return np.array([self.cumulative_total]), -10, True, False, {}

        self.cumulative_total += chosen_number

        if self.cumulative_total == 23:
            # Current player wins
            self.done = True
            return np.array([self.cumulative_total]), 1, True, False, {}
        elif self.cumulative_total > 23:
            # Current player loses by exceeding 23
            self.done = True
            return np.array([self.cumulative_total]), -10, True, False, {}
        else:
            # Valid move, switch to the other player
            self.current_player *= -1
            return np.array([self.cumulative_total]), 0, False, False, {}

    def render(self):
        # Provide a text representation of the current state
        return f"Current Total is {self.cumulative_total}."

    def valid_moves(self):
        # Return a list of valid actions based on the current total
        return [
            action for action in range(4) if self.cumulative_total + (action + 1) <= 23
        ]
