import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Two actions: 0 for 'A', 1 for 'B'
        self.action_space = spaces.Discrete(2)
        # Observation space: Fixed-size array representing the sequence
        # -1 for padding, 0 for 'A', 1 for 'B'
        self.max_seq_len = 50  # Maximum sequence length
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.max_seq_len,), dtype=np.int8
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = []
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Append the action to the sequence
        self.sequence.append(action)

        # Check for the target pattern 'ABBA' which corresponds to [0,1,1,0]
        if self._check_for_pattern():
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch the current player
        self.current_player = 1 if self.current_player == 2 else 2

        return self._get_observation(), 0, False, False, {}

    def render(self):
        symbol_map = {0: "A", 1: "B"}
        sequence_str = "".join(symbol_map.get(s, "?") for s in self.sequence)
        return f"Shared Sequence: {sequence_str}"

    def valid_moves(self):
        # Both 'A' and 'B' are always valid moves
        return [0, 1]

    def _get_observation(self):
        # Create observation array with padding
        obs = np.full(self.max_seq_len, -1, dtype=np.int8)
        seq_len = len(self.sequence)
        if seq_len > self.max_seq_len:
            # Truncate the sequence if it exceeds maximum length
            obs[:] = self.sequence[-self.max_seq_len :]
        else:
            obs[:seq_len] = self.sequence
        return obs

    def _check_for_pattern(self):
        # Define the target pattern as a list
        target_pattern = [0, 1, 1, 0]  # 'A', 'B', 'B', 'A'
        seq_len = len(self.sequence)
        # Only check if sequence is at least as long as the target pattern
        if seq_len >= len(target_pattern):
            # Scan through the sequence to check for the pattern
            for i in range(seq_len - len(target_pattern) + 1):
                if self.sequence[i : i + len(target_pattern)] == target_pattern:
                    return True
        return False
