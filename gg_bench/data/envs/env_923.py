import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=10):
        super(CustomEnv, self).__init__()

        self.N = N
        # Define action and observation space
        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.N,), dtype=np.int8
        )

        # Initialize variables
        self.bits = np.zeros(self.N, dtype=np.int8)
        self.current_player = 1
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bits = np.zeros(self.N, dtype=np.int8)
        self.current_player = 1
        self.done = False
        return self.bits.copy(), {}  # Return observation and info

    def is_action_valid(self, action):
        if self.bits[action] == 1:
            return False
        if action > 0 and self.bits[action - 1] == 1:
            return False
        if action < self.N - 1 and self.bits[action + 1] == 1:
            return False
        return True

    def valid_moves(self):
        valid_positions = []
        for pos in range(self.N):
            if self.is_action_valid(pos):
                valid_positions.append(pos)
        return valid_positions

    def step(self, action):
        if self.done or not self.is_action_valid(action):
            self.done = True
            return self.bits.copy(), -10, True, False, {}
        # Flip the bit
        self.bits[action] = 1

        # Check if the next player has any valid moves
        if len(self.valid_moves()) == 0:
            self.done = True
            return self.bits.copy(), 1, True, False, {}
        else:
            # Switch to the next player
            self.current_player *= -1
            return self.bits.copy(), 0, False, False, {}

    def render(self):
        positions_str = "Positions: " + " ".join(f"[{i}]" for i in range(self.N))
        bits_str = "Bits:      " + " ".join(f"[{self.bits[i]}]" for i in range(self.N))
        return positions_str + "\n" + bits_str
