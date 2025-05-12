import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, N_start=30):
        super(CustomEnv, self).__init__()
        # Game parameters
        self.N_start = N_start
        self.N_max = self.N_start

        # Define action_space and observation_space
        # Actions are integers from 0 to N_max inclusive
        # Valid D values are between 2 and N-1
        self.action_space = spaces.Discrete(self.N_max + 1)
        self.observation_space = spaces.Box(
            low=0, high=self.N_max, shape=(1,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Optionally set a new starting N
        if options and "N_start" in options:
            self.N_start = options["N_start"]
            self.N_max = self.N_start
            self.action_space = spaces.Discrete(self.N_max + 1)
            self.observation_space = spaces.Box(
                low=0, high=self.N_max, shape=(1,), dtype=np.int32
            )

        self.N = self.N_start
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        D = action
        reward = 0
        terminated = False
        truncated = False

        # Check for invalid action
        if D <= 1 or D >= self.N or self.N % D != 0:
            reward = -10  # Penalty for invalid move
            self.done = True
            terminated = True
            return np.array([self.N], dtype=np.int32), reward, terminated, truncated, {}

        # Valid action: subtract D from N
        self.N -= D

        # Check for win condition
        if self.N == 0:
            reward = 1  # Current player wins
            self.done = True
            terminated = True
            return np.array([self.N], dtype=np.int32), reward, terminated, truncated, {}

        # Check if the next player has any valid moves
        valid_divisors = [d for d in range(2, self.N) if self.N % d == 0]
        if not valid_divisors:
            reward = 1  # Current player wins as the opponent has no moves
            self.done = True
            terminated = True
        else:
            # Game continues; switch to the next player
            self.current_player = 3 - self.current_player  # Toggles between 1 and 2

        return np.array([self.N], dtype=np.int32), reward, terminated, truncated, {}

    def render(self):
        return f"Current N: {self.N}"

    def valid_moves(self):
        # Returns a list of valid actions (divisors) for the current N
        return [D for D in range(2, self.N) if self.N % D == 0]
