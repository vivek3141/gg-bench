import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0-19
        # Actions 0-9: Subtract 1-10
        # Actions 10-19: Add 1-10
        self.action_space = spaces.Discrete(20)

        # Observation space: shared number between 0 and 40 (inclusive)
        self.observation_space = spaces.Box(low=0, high=40, shape=(1,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = 20  # Starting shared number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.shared_number]), {}  # Observation and info

    def step(self, action):
        if self.done:
            return np.array([self.shared_number]), 0, True, False, {}

        # Interpret the action
        if action < 0 or action >= 20:
            # Invalid action index
            reward = -10
            self.done = True
            return np.array([self.shared_number]), reward, True, False, {}

        if action < 10:
            number = action + 1  # Subtract numbers from 1 to 10
            new_shared_number = self.shared_number - number
        else:
            number = action - 9  # Add numbers from 1 to 10
            new_shared_number = self.shared_number + number

        # Check validity of the move
        if new_shared_number < 0 or new_shared_number > 40:
            # Invalid move
            reward = -10
            self.done = True
            return np.array([self.shared_number]), reward, True, False, {}

        # Update shared number
        self.shared_number = new_shared_number

        # Check for victory
        if self.shared_number == 0:
            reward = 1  # Current player wins
            self.done = True
            return np.array([self.shared_number]), reward, True, False, {}
        else:
            reward = 0  # Valid move, game continues
            self.current_player = 3 - self.current_player  # Switch players
            return np.array([self.shared_number]), reward, False, False, {}

    def render(self):
        # Return a string representation of the current state
        return f"Current Shared Number: {self.shared_number}\nPlayer {self.current_player}'s Turn"

    def valid_moves(self):
        valid_actions = []
        for action in range(20):
            if action < 10:
                number = action + 1
                potential_number = self.shared_number - number
            else:
                number = action - 9
                potential_number = self.shared_number + number

            if 0 <= potential_number <= 40:
                valid_actions.append(action)
        return valid_actions
