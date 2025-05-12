import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, starting_number=100):
        super(CustomEnv, self).__init__()

        self.N_start = starting_number
        self.N = self.N_start
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Define action and observation space
        # Action space: Discrete actions from 0 to N_start (inclusive)
        # Action 1: Subtract 1
        # Actions 2 to N_start: Attempt to divide N by action index
        self.action_space = spaces.Discrete(self.N_start + 1)

        # Observation space: The current value of N
        self.observation_space = spaces.Box(
            low=1, high=self.N_start, shape=(1,), dtype=np.int64
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_start
        self.current_player = 1  # Player 1 starts
        self.done = False
        obs = np.array([self.N], dtype=np.int64)
        return obs, {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.N], dtype=np.int64),
                0,
                True,
                False,
                {},
            )

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return (
                np.array([self.N], dtype=np.int64),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        if action == 1:
            # Subtract 1
            self.N -= 1
        else:
            # Divide N by action index
            self.N = self.N // action

        if self.N == 1:
            self.done = True
            return (
                np.array([self.N], dtype=np.int64),
                1,
                True,
                False,
                {},
            )

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1
        return (
            np.array([self.N], dtype=np.int64),
            0,
            False,
            False,
            {},
        )

    def render(self):
        return f"Current N: {self.N}, Current Player: {self.current_player}"

    def valid_moves(self):
        if self.N <= 1:
            return []
        elif self.is_prime(self.N):
            return [1]  # Only action is to subtract 1
        else:
            # Proper divisors greater than 1 and less than N
            divisors = [i for i in range(2, self.N) if self.N % i == 0]
            return divisors

    @staticmethod
    def is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
