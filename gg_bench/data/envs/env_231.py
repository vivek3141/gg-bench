import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: append 0 or 1
        self.action_space = spaces.Discrete(2)
        # Observation space: array of length 20, with values -1 (empty), 0, or 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = np.full(
            20, -1, dtype=np.int8
        )  # Initialize with -1 indicating empty positions
        self.seq_length = 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.sequence.copy(), {}

    def step(self, action):
        if self.done:
            # Game is over
            return self.sequence.copy(), -10, True, False, {}

        if action not in [0, 1]:
            # Invalid action
            self.done = True
            reward = -10  # Penalize for invalid action
            return self.sequence.copy(), reward, True, False, {}

        if self.seq_length >= 20:
            # Maximum length reached, current player loses
            self.done = True
            reward = -10  # Current player loses
            return self.sequence.copy(), reward, True, False, {}

        # Append the action to the sequence
        self.sequence[self.seq_length] = action
        self.seq_length += 1

        # Check winning condition: binary number divisible by 3
        binary_str = "".join(str(bit) for bit in self.sequence[: self.seq_length])
        binary_number = int(binary_str, 2)
        if binary_number % 3 == 0:
            # Winning condition satisfied
            self.done = True
            reward = 1  # Current player wins
            return self.sequence.copy(), reward, True, False, {}

        # Check if maximum length is reached after this move
        if self.seq_length >= 20:
            self.done = True
            reward = -10  # Current player loses
            return self.sequence.copy(), reward, True, False, {}

        # Switch player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        # Valid move, no win yet
        reward = -10
        return self.sequence.copy(), reward, False, False, {}

    def render(self):
        binary_str = "".join(str(bit) for bit in self.sequence[: self.seq_length])
        return f"Current Sequence: {binary_str}"

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1]
