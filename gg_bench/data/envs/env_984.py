import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define maximum allowed starting number
        self.N_MAX = 1000  # Maximum N_start supported

        # Default starting number
        self.N_start = 16

        # Initialize current number
        self.current_number = self.N_start

        # Define action space
        # Actions correspond to selecting integers from 2 up to N_MAX - 1
        # Action indices range from 0 to N_MAX - 3, mapping to divisors from 2 to N_MAX - 1
        self.action_space = spaces.Discrete(
            self.N_MAX - 2
        )  # actions from 0 to N_MAX - 3

        # Define observation space
        self.observation_space = spaces.Box(
            low=1, high=self.N_MAX, shape=(1,), dtype=np.int32
        )

        # Initialize current player (1 or 2)
        self.current_player = 1

        # Game over flag
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Allow starting number to be set via options
        if options is not None and "N_start" in options:
            self.current_number = min(options["N_start"], self.N_MAX)
        else:
            self.current_number = self.N_start
        # Ensure starting number is valid
        if self.current_number < 2:
            raise ValueError("Starting number must be greater than 1.")
        self.current_player = 1
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Ensure action is within valid range
        if not self.action_space.contains(action):
            self.done = True
            reward = -10  # Invalid action
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Map action index to actual divisor: divisor = action + 2
        divisor = action + 2

        # Check if the action is a proper divisor of the current number
        if (
            self.current_number % divisor != 0
            or divisor <= 1
            or divisor >= self.current_number
        ):
            self.done = True
            reward = -10  # Invalid move, player loses
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move: update the current number
        self.current_number = divisor

        # Check winning condition
        if self.current_number == 1:
            self.done = True
            reward = 1  # Current player wins
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        # No immediate reward
        reward = 0
        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        return f"Current number is {self.current_number}. It's Player {self.current_player}'s turn."

    def valid_moves(self):
        # Get all valid proper divisors of the current number
        divisors = [
            i for i in range(2, self.current_number) if self.current_number % i == 0
        ]
        # Map divisors to action indices: action = divisor - 2
        actions = [d - 2 for d in divisors]
        return actions
