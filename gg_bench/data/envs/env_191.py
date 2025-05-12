import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: numbers from 2 to 50 (indices 0 to 48)
        self.action_space = spaces.Discrete(49)
        # Observation space: pool represented by 0 (removed) or 1 (available)
        self.observation_space = spaces.Box(low=0, high=1, shape=(49,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the number pool with all numbers available
        self.pool = np.ones(49, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.pool.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.pool.copy(), 0, True, False, {}

        # Validate action
        if action < 0 or action >= 49 or self.pool[action] == 0:
            # Invalid action: number not in pool or out of bounds
            self.done = True
            return self.pool.copy(), -10, True, False, {}

        # Valid action: remove the selected number and its multiples
        n = action + 2  # Convert index to actual number
        multiple = n
        while multiple <= 50:
            idx = multiple - 2  # Convert number back to index
            self.pool[idx] = 0
            multiple += n

        # Check if the opponent has any valid moves
        if not np.any(self.pool):
            # Opponent has no valid moves; current player wins
            self.done = True
            return self.pool.copy(), 1, True, False, {}
        else:
            # Switch to the next player
            self.current_player = 3 - self.current_player
            return self.pool.copy(), 0, False, False, {}

    def render(self):
        # Visual representation of the current pool
        remaining_numbers = [str(i + 2) for i in range(49) if self.pool[i] == 1]
        board_str = "Remaining Numbers:\n" + ", ".join(remaining_numbers)
        return board_str

    def valid_moves(self):
        # Return a list of valid action indices
        return [i for i in range(49) if self.pool[i] == 1]
