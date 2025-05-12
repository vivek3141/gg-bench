import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=10):
        super(CustomEnv, self).__init__()

        self.N = N  # Length of the binary string
        self.action_space = spaces.Discrete(2 * self.N - 1)

        # Observation space is the binary string of length N,
        # with values 0 or 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.N,), dtype=np.int8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.N, dtype=np.int8)
        self.current_player = 1
        self.done = False
        return self.state.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, return current state
            return self.state.copy(), 0, True, False, {}

        positions_to_flip = []

        if 0 <= action < self.N:
            # Flip single zero at position 'action'
            positions_to_flip = [action]
        elif self.N <= action < 2 * self.N - 1:
            pos = action - self.N
            # Flip zeros at positions 'pos' and 'pos + 1'
            positions_to_flip = [pos, pos + 1]
        else:
            # Invalid action index (should not happen)
            self.done = True
            return self.state.copy(), -10, True, False, {}

        # Check if positions to flip are valid
        for pos in positions_to_flip:
            if pos < 0 or pos >= self.N or self.state[pos] != 0:
                # Invalid move
                self.done = True
                return self.state.copy(), -10, True, False, {}

        # Flip the bits
        self.state[positions_to_flip] = 1

        # Check for win condition
        if np.all(self.state == 1):
            self.done = True
            return self.state.copy(), 1, True, False, {}

        # No win yet, switch player
        self.current_player = 2 if self.current_player == 1 else 1

        return self.state.copy(), 0, False, False, {}

    def render(self):
        positions = "Positions: " + " ".join(str(i + 1) for i in range(self.N)) + "\n"
        values = "Values:    " + " ".join(str(self.state[i]) for i in range(self.N))
        board_str = positions + values
        print(board_str)

    def valid_moves(self):
        valid_actions = []

        # Single flips
        for pos in range(self.N):
            if self.state[pos] == 0:
                valid_actions.append(pos)

        # Double flips
        for pos in range(self.N - 1):
            if self.state[pos] == 0 and self.state[pos + 1] == 0:
                action = self.N + pos
                valid_actions.append(action)

        return valid_actions
