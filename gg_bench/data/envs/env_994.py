import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete actions from 0 to 7, corresponding to numbers 2 to 9
        self.action_space = spaces.Discrete(8)

        # Observation space: Running total, an integer from 1 to 100
        self.observation_space = spaces.Box(low=1, high=100, shape=(1,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.running_total = 1
        self.done = False
        self.current_player = 1  # Player 1 starts; for self-play, this can be ignored
        return np.array([self.running_total], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # If the game is already over, return the current state
            return (
                np.array([self.running_total], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Map action (0-7) to chosen number (2-9)
        chosen_number = action + 2

        # Calculate the new running total
        next_running_total = self.running_total * chosen_number

        if next_running_total == 100:
            # Current player wins
            reward = 1
            self.done = True
            info = {}
            return (
                np.array([next_running_total], dtype=np.int32),
                reward,
                True,
                False,
                info,
            )
        elif next_running_total > 100:
            # Current player loses
            reward = -10
            self.done = True
            info = {}
            return (
                np.array([next_running_total], dtype=np.int32),
                reward,
                True,
                False,
                info,
            )
        else:
            # Continue the game
            self.running_total = next_running_total
            reward = 0  # No reward for a regular move
            self.current_player *= -1  # Switch current player
            info = {}
            return (
                np.array([self.running_total], dtype=np.int32),
                reward,
                False,
                False,
                info,
            )

    def render(self):
        return f"Current running total: {self.running_total}"

    def valid_moves(self):
        valid_actions = []
        for action in range(8):
            chosen_number = action + 2
            if self.running_total * chosen_number <= 100:
                valid_actions.append(action)
        return valid_actions
