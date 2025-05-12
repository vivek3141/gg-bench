import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: multipliers from 2 to 9 inclusive
        self.action_space = spaces.Discrete(
            8
        )  # Actions: indices from 0 to 7, corresponding to multipliers 2-9

        # Define observation space: players' totals
        # Totals can range from 1 to a maximum of 1000 for safety
        self.observation_space = spaces.Box(
            low=1, high=1000, shape=(2,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.totals = [1, 1]  # Player 1 and Player 2 totals start at 1
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return np.array(self.totals, dtype=np.int32), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return np.array(self.totals, dtype=np.int32), 0, True, False, {}

        # Validate action
        if not self.action_space.contains(action):
            reward = -10
            self.done = True
            return np.array(self.totals, dtype=np.int32), reward, True, False, {}

        # Map action to multiplier (2 to 9)
        multiplier = action + 2  # Action 0 corresponds to multiplier 2

        # Update current player's total
        self.totals[self.current_player] *= multiplier
        curr_total = self.totals[self.current_player]

        # Check for win condition
        if curr_total == 100:
            reward = 1
            self.done = True
            return np.array(self.totals, dtype=np.int32), reward, True, False, {}

        # Check for loss condition
        if curr_total > 100:
            reward = -10
            self.done = True
            return np.array(self.totals, dtype=np.int32), reward, True, False, {}

        # Continue game
        reward = 0
        self.current_player = 1 - self.current_player  # Switch player
        return np.array(self.totals, dtype=np.int32), reward, False, False, {}

    def render(self):
        return (
            f"Player 1 Total: {self.totals[0]}, "
            f"Player 2 Total: {self.totals[1]}, "
            f"Current Turn: Player {self.current_player + 1}"
        )

    def valid_moves(self):
        # All moves are valid; players must choose a multiplier between 2 and 9
        return list(range(self.action_space.n))
