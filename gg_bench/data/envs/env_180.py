import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 10 possible pairs of indices (0-9 mapping to pairs of indices in the pool)
        self.action_space = spaces.Discrete(10)

        # Observation space: Array of 5 numbers (pool), with possible numbers -1 (empty), 1-15
        self.observation_space = spaces.Box(low=-1, high=15, shape=(5,), dtype=np.int32)

        # Mapping from action index to pairs of indices in the pool
        self.action_to_indices = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        ]

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the shared pool
        self.pool = [1, 2, 3, 4, 5]
        # Fill up to 5 elements with -1 if necessary
        self.pool += [-1] * (5 - len(self.pool))
        self.pool = np.array(self.pool, dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.pool.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.pool.copy(), -10, True, False, {}
        # Map action to indices
        if action < 0 or action >= 10:
            # Invalid action index
            return self.pool.copy(), -10, True, False, {}

        i, j = self.action_to_indices[action]

        # Check if indices are valid (numbers at positions are not -1)
        if self.pool[i] == -1 or self.pool[j] == -1:
            # Invalid move
            return self.pool.copy(), -10, True, False, {}

        num1 = self.pool[i]
        num2 = self.pool[j]

        # Remove the numbers from the pool
        self.pool[i] = -1
        self.pool[j] = -1

        # Compute the sum
        new_number = num1 + num2

        # Prepare info for rendering
        info = {
            "player": self.current_player,
            "merged_numbers": (num1, num2),
            "new_number": new_number,
            "action": action,
            "pool_before": self.pool.copy(),
        }

        # If sum is less than or equal to 15, add it to the pool
        if new_number <= 15:
            # Find the first empty slot
            for idx in range(5):
                if self.pool[idx] == -1:
                    self.pool[idx] = new_number
                    break  # Added the new number
            else:
                # This should not happen as pool cannot have more than 5 numbers
                pass  # For safety
            # Check for win condition
            if new_number == 15:
                self.done = True
                return self.pool.copy(), 1, True, False, info

        # Check if after move, pool has fewer than 2 numbers
        available_numbers = [num for num in self.pool if num != -1]
        if len(available_numbers) < 2:
            # Game cannot continue, current player loses
            self.done = True
            return self.pool.copy(), 0, True, False, info

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        return self.pool.copy(), -10, False, False, info

    def render(self):
        pool_str = f"Shared Pool: {self.get_pool_numbers()}\n"
        player_str = f"Player {self.current_player}'s turn."
        return pool_str + player_str

    def get_pool_numbers(self):
        return [num for num in self.pool if num != -1]

    def valid_moves(self):
        valid_actions = []
        for idx, (i, j) in enumerate(self.action_to_indices):
            if self.pool[i] != -1 and self.pool[j] != -1:
                valid_actions.append(idx)
        return valid_actions
