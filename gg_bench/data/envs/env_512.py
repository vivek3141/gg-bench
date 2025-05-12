import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, initial_N=25):
        super(CustomEnv, self).__init__()

        self.initial_N = initial_N

        # Actions are digits from 1 to 9 (indices 0 to 8)
        self.action_space = spaces.Discrete(9)

        # Observation is the current N
        self.observation_space = spaces.Box(
            low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "N" in options:
            self.N = options["N"]
        else:
            self.N = self.initial_N
        self.current_player = 1  # Player 1 starts
        self.done = False

        return np.array([self.N], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        valid_moves = self.valid_moves()
        if len(valid_moves) == 0:
            # Current player cannot move, they lose
            self.done = True
            # Opponent wins, reward -1 for current player
            return np.array([self.N], dtype=np.int32), -1, True, False, {}

        if action not in valid_moves:
            # Invalid move
            self.done = True
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        digit = action + 1  # Map action index to digit

        if self.N - digit < 0:
            # Move results in negative N, invalid
            self.done = True
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        self.N -= digit

        if self.N == 0:
            # Current player wins
            self.done = True
            return np.array([self.N], dtype=np.int32), 1, True, False, {}

        # Switch player
        self.current_player *= -1

        # Check if next player has valid moves
        valid_moves_next = self.valid_moves()
        if len(valid_moves_next) == 0:
            # Next player cannot move, current player wins
            self.done = True
            return np.array([self.N], dtype=np.int32), 1, True, False, {}

        return np.array([self.N], dtype=np.int32), 0, False, False, {}

    def render(self):
        s = f"Current N: {self.N}\n"
        digits = [int(d) for d in str(self.N)]
        valid_digits = set(digits) - set([0])
        s += f"Available digits to subtract: {', '.join(map(str, sorted(valid_digits)))}\n"
        s += f"Player {1 if self.current_player == 1 else 2}'s turn.\n"
        return s

    def valid_moves(self):
        if self.done:
            return []

        digits = [int(d) for d in str(self.N)]
        valid_digits = set(digits) - set([0])

        valid_actions = []
        for d in valid_digits:
            if self.N - d >= 0:
                valid_actions.append(d - 1)  # Map digit to action index
        return valid_actions
