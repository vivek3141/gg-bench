import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_value=1000):
        super(CustomEnv, self).__init__()

        self.target_value = target_value

        # Define action space: multipliers from 2 to 9 (actions 0 to 7 correspond to multipliers 2 to 9)
        self.action_space = spaces.Discrete(8)

        # Define observation space: cumulative number (scalar)
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([self.target_value]), dtype=np.int64
        )

        # Initialize random number generator
        self.np_random = np.random.RandomState()

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            self.np_random.seed(seed)
        self.cumulative_number = 1
        self.done = False

        return np.array([self.cumulative_number]), {}  # Observation and info

    def step(self, action):
        if self.done:
            return np.array([self.cumulative_number]), 0, True, False, {}

        # Map action to multiplier (0 -> 2, 1 -> 3, ..., 7 -> 9)
        multiplier = action + 2  # Actions 0-7 correspond to multipliers 2-9

        # Check if action is valid
        if action not in self.valid_moves():
            self.done = True
            reward = -10  # Invalid action
            return np.array([self.cumulative_number]), reward, True, False, {}

        # Player's move
        self.cumulative_number *= multiplier

        # Check if player loses
        if self.cumulative_number >= self.target_value:
            self.done = True
            reward = -1  # Player loses
            return np.array([self.cumulative_number]), reward, True, False, {}

        # Game continues, simulate opponent's move
        opponent_action = self.opponent_policy()
        opponent_multiplier = opponent_action + 2
        self.cumulative_number *= opponent_multiplier

        # Check if opponent loses (player wins)
        if self.cumulative_number >= self.target_value:
            self.done = True
            reward = 1  # Player wins
            return np.array([self.cumulative_number]), reward, True, False, {}

        # Game continues without termination
        reward = -10  # Penalty for valid move
        return np.array([self.cumulative_number]), reward, False, False, {}

    def opponent_policy(self):
        # Opponent selects a random valid action
        valid_actions = self.valid_moves()
        return self.np_random.choice(valid_actions)

    def valid_moves(self):
        # Valid actions correspond to multipliers 2 to 9 (actions 0 to 7)
        return list(range(8))

    def render(self):
        # Return a string representation of the current state
        return f"Cumulative Number: {self.cumulative_number}"
