import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        # There are 15 possible actions: 5 flip actions and 10 swap actions
        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.int32)

        # Precompute swap action pairs (unordered combinations)
        self.swap_pairs = [
            (0, 1),  # Positions 1 and 2
            (0, 2),  # Positions 1 and 3
            (0, 3),  # Positions 1 and 4
            (0, 4),  # Positions 1 and 5
            (1, 2),  # Positions 2 and 3
            (1, 3),  # Positions 2 and 4
            (1, 4),  # Positions 2 and 5
            (2, 3),  # Positions 3 and 4
            (2, 4),  # Positions 3 and 5
            (3, 4),  # Positions 4 and 5
        ]

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the binary string to all zeros
        self.binary_string = np.zeros(5, dtype=np.int32)

        # Set the current player (1 or -1)
        self.current_player = 1

        # Game over flag
        self.done = False

        return self.binary_string.copy(), {}  # Observation and info

    def step(self, action):
        if self.done:
            # If the game has ended, return the current state
            return self.binary_string.copy(), 0, True, False, {}

        # Perform the action
        if action < 5:
            # Flip action (actions 0-4 correspond to positions 0-4)
            position = action
            # Flip the bit at the specified position
            self.binary_string[position] = 1 - self.binary_string[position]
        else:
            # Swap action
            swap_index = action - 5
            pos1, pos2 = self.swap_pairs[swap_index]
            # Swap the bits at the specified positions
            self.binary_string[pos1], self.binary_string[pos2] = (
                self.binary_string[pos2],
                self.binary_string[pos1],
            )

        # Check for victory condition
        if np.all(self.binary_string == 1):
            self.done = True
            reward = 1  # Current player wins
            return self.binary_string.copy(), reward, True, False, {}

        # Valid move made, but game continues
        reward = -10

        # Switch to the other player
        self.current_player *= -1

        return self.binary_string.copy(), reward, False, False, {}

    def render(self):
        # Return a string representation of the binary string
        binary_str = " ".join(map(str, self.binary_string))
        return f"Binary String: {binary_str}"

    def valid_moves(self):
        # All actions from 0 to 14 are valid
        return list(range(15))
