import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define constants
        self.initial_N = 30  # Starting number (N)
        self.MAX_N = 100  # Maximum allowed value for N

        # Define action and observation space
        # The action space consists of integers from 0 to MAX_N inclusive
        self.action_space = spaces.Discrete(self.MAX_N + 1)

        # The observation is the current value of N
        self.observation_space = spaces.Box(
            low=1, high=self.MAX_N, shape=(1,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.initial_N
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        action = int(action)  # Ensure the action is an integer

        # Check if the game is already over
        if self.done:
            return (
                np.array([self.N], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Validate the action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return (
                np.array([self.N], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )  # Invalid move

        # Update N based on the chosen proper divisor
        self.N = self.N // action

        # Check for win condition
        if self.N == 1:
            self.done = True
            return (
                np.array([self.N], dtype=np.int32),
                1,
                True,
                False,
                {},
            )  # Current player wins
        else:
            # Switch turns
            self.current_player = 2 if self.current_player == 1 else 1
            return (
                np.array([self.N], dtype=np.int32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        return (
            f"Current N: {self.N}, "
            f"Player {self.current_player}'s turn. "
            f"Valid moves: {self.valid_moves()}"
        )

    def valid_moves(self):
        if self.N <= 1:
            return []
        # Proper divisors are integers greater than 1 and less than N that divide N exactly
        proper_divisors = [
            i for i in range(2, self.MAX_N + 1) if i < self.N and self.N % i == 0
        ]
        return proper_divisors
