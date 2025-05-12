import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.numbers_state = np.ones(9, dtype=np.float32)
        self.current_player = 1
        self.done = False
        return self.numbers_state.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done or self.numbers_state[action] == 0:
            # Invalid move or game is already over
            return self.numbers_state.copy(), -10, True, False, {}

        n = action + 1  # Number selected by the player (1-9)
        # Remove the selected number and its multiples
        for i in range(9):
            num = i + 1
            if num % n == 0 and self.numbers_state[i] == 1:
                self.numbers_state[i] = 0

        # Check if any numbers are left
        if np.sum(self.numbers_state) == 0:
            # Current player wins
            self.done = True
            return self.numbers_state.copy(), 1, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        return self.numbers_state.copy(), -10, False, False, {}

    def render(self):
        grid = ""
        for i in range(3):
            row = ""
            for j in range(3):
                idx = i * 3 + j
                num = idx + 1
                if self.numbers_state[idx] == 1:
                    row += f"{num} "
                else:
                    row += "X "
            grid += row.strip() + "\n"
        return grid.strip()

    def valid_moves(self):
        return [i for i in range(9) if self.numbers_state[i] == 1]
