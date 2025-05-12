import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=60):
        super(CustomEnv, self).__init__()

        # Initial number N (default is 60)
        self.initial_N = N

        # Maximum possible N (adjusted as needed)
        self.N_max = 100

        # Define action space (possible subtractions from N)
        # Actions are integers from 0 to N_max
        self.action_space = spaces.Discrete(self.N_max + 1)

        # Define observation space (current value of N)
        self.observation_space = spaces.Box(
            low=1, high=self.N_max, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "N" in options:
            self.N = options["N"]
        else:
            self.N = self.initial_N  # Start with initial N
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Observation and info

    def get_proper_divisors(self, n):
        # Returns proper divisors of n excluding 1 and n
        divisors = [i for i in range(2, n) if n % i == 0]
        return divisors

    def valid_moves(self):
        # Returns list of valid actions (proper divisors of current N)
        return self.get_proper_divisors(self.N)

    def step(self, action):
        if self.done:
            return (
                np.array([self.N], dtype=np.int32),
                0,
                True,
                False,
                {},  # Observation, reward, terminated, truncated, info
            )

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Valid move: subtract action from N
        self.N -= action

        # Check if opponent can make a move
        next_valid_moves = self.get_proper_divisors(self.N)
        if len(next_valid_moves) == 0:
            # Opponent cannot move, current player wins
            self.done = True
            reward = 1
            return np.array([self.N], dtype=np.int32), reward, True, False, {}
        else:
            # Game continues, switch player
            self.current_player *= -1
            reward = 0
            return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def render(self):
        return f"Current N: {self.N}, Player: {self.current_player}"
