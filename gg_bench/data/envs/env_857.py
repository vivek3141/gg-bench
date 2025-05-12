import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        # Actions are integers from 0 to 5 (inclusive), corresponding to additions of 1 to 6
        self.action_space = spaces.Discrete(6)

        # Observation is the current total, integer between 0 and 30 inclusive
        self.observation_space = spaces.Box(low=0, high=30, shape=(1,), dtype=np.int32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_total = 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array([self.current_total], dtype=np.int32)
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, return current state
            observation = np.array([self.current_total], dtype=np.int32)
            return (
                observation,
                0,
                True,
                False,
                {},
            )  # observation, reward, terminated, truncated, info

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action or action causes total to exceed 30
            self.done = True
            observation = np.array([self.current_total], dtype=np.int32)
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # observation, reward, terminated, truncated, info

        # Apply the action
        addition = action + 1  # Map action index to addition (1 to 6)
        self.current_total += addition

        observation = np.array([self.current_total], dtype=np.int32)

        # Check for win or lose conditions
        if self.current_total == 30:
            # Current player wins
            self.done = True
            return (
                observation,
                1,
                True,
                False,
                {},
            )  # observation, reward, terminated, truncated, info
        elif self.current_total > 30:
            # Current player loses
            self.done = True
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # observation, reward, terminated, truncated, info
        else:
            # Game continues
            self.current_player = 2 if self.current_player == 1 else 1  # Switch player
            return (
                observation,
                0,
                False,
                False,
                {},
            )  # observation, reward, terminated, truncated, info

    def render(self):
        return (
            f"Current Total: {self.current_total}, Player {self.current_player}'s turn."
        )

    def valid_moves(self):
        # Returns a list of valid action indices (0 to 5)
        valid_actions = []
        for action in range(6):
            addition = action + 1
            if self.current_total + addition <= 30:
                valid_actions.append(action)
        return valid_actions
