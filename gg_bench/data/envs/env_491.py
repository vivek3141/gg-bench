import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=9):
        super(CustomEnv, self).__init__()
        assert N % 2 == 1 and N > 1, "N must be an odd integer greater than 1."
        self.N = N  # The length of the sequence
        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Box(
            low=0, high=self.N, shape=(self.N,), dtype=np.int32
        )
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = np.arange(1, self.N + 1, dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.sequence.copy(), {}  # Return observation and info

    def step(self, action):
        info = {}
        truncated = False  # No truncation in this game

        # Check if the game has already ended
        if self.done:
            return self.sequence.copy(), 0, True, truncated, info

        # Validate the action
        if action < 0 or action >= self.N or self.sequence[action] == 0:
            # Invalid move
            reward = -10
            self.done = True
            return self.sequence.copy(), reward, True, truncated, info

        # Valid move: remove the selected number and its adjacent numbers
        self.sequence[action] = 0  # Remove selected number
        # Remove left neighbor if it exists
        if action > 0 and self.sequence[action - 1] != 0:
            self.sequence[action - 1] = 0
        # Remove right neighbor if it exists
        if action < self.N - 1 and self.sequence[action + 1] != 0:
            self.sequence[action + 1] = 0

        # Check if any numbers remain
        if not np.any(self.sequence):
            # Current player wins
            reward = 1
            self.done = True
        else:
            # Game continues
            reward = 0
            # Switch to the other player
            self.current_player = 2 if self.current_player == 1 else 1

        return self.sequence.copy(), reward, self.done, truncated, info

    def render(self):
        # Return a string representation of the current sequence
        sequence_str = ""
        for num in self.sequence:
            if num == 0:
                sequence_str += "_ "
            else:
                sequence_str += f"{num} "
        return sequence_str.strip()

    def valid_moves(self):
        # Return a list of valid moves (indices of available numbers)
        return [i for i in range(self.N) if self.sequence[i] != 0]
