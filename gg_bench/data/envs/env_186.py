import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 8 possible actions: swapping positions 0 & 1, 1 & 2, ..., 7 & 8
        self.action_space = spaces.Discrete(8)

        # Observation space: sequence of numbers from 1 to 9
        self.observation_space = spaces.Box(low=1, high=9, shape=(9,), dtype=np.int32)

        # Initialize the sequence and game status
        self.sequence = None
        self.current_player = 1
        self.done = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create a shuffled sequence of numbers from 1 to 9
        self.sequence = np.arange(1, 10)
        self.np_random.shuffle(self.sequence)

        self.current_player = 1
        self.done = False

        return self.sequence.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.sequence.copy(), -10, True, False, {}

        # Check for valid action
        if not (0 <= action < 8):
            self.done = True
            return self.sequence.copy(), -10, True, False, {}

        # Swap the numbers at the selected adjacent positions
        self.sequence[action], self.sequence[action + 1] = (
            self.sequence[action + 1],
            self.sequence[action],
        )

        # Check if the sequence is fully sorted
        if np.all(self.sequence[:-1] <= self.sequence[1:]):
            self.done = True
            return self.sequence.copy(), 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        return self.sequence.copy(), -10, False, False, {}

    def render(self):
        positions = "Positions: " + " ".join(str(i + 1) for i in range(9)) + "\n"
        numbers = "Numbers:   " + " ".join(str(num) for num in self.sequence) + "\n"
        return positions + numbers

    def valid_moves(self):
        if self.done:
            return []
        else:
            return list(range(8))  # Positions 0 to 7 correspond to possible swaps
