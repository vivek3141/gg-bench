import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 for Add 2, 1 for Multiply by 2
        self.action_space = spaces.Discrete(2)

        # Define observation space: current total
        self.observation_space = spaces.Box(low=1, high=128, shape=(1,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.total = 1
        self.current_player = 1  # Player 1 or Player 2
        self.done = False

        return np.array([self.total], dtype=np.int32), {}

    def step(self, action):
        if action not in [0, 1]:
            # Invalid action index
            return np.array([self.total], dtype=np.int32), -10, True, False, {}

        if self.done:
            # Game is already over
            return np.array([self.total], dtype=np.int32), -10, True, False, {}

        # Apply action
        if action == 0:
            new_total = self.total + 2
        else:  # action == 1
            new_total = self.total * 2

        # Check for exceeding 64
        if new_total > 64:
            # Current player loses
            self.total = new_total
            self.done = True
            reward = -10
            return np.array([self.total], dtype=np.int32), reward, True, False, {}

        # Check for win
        if new_total == 64:
            # Current player wins
            self.total = new_total
            self.done = True
            reward = 1
            return np.array([self.total], dtype=np.int32), reward, True, False, {}

        # Game continues
        self.total = new_total
        # Switch players
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0
        return np.array([self.total], dtype=np.int32), reward, False, False, {}

    def render(self):
        return (
            f"Current total: {self.total}, Current player: Player {self.current_player}"
        )

    def valid_moves(self):
        # Return list of valid actions that do not cause total to exceed 64
        valid_actions = []
        if self.total + 2 <= 64:
            valid_actions.append(0)
        if self.total * 2 <= 64:
            valid_actions.append(1)
        return valid_actions
