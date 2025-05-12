import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=12, max_number=100):
        super(CustomEnv, self).__init__()

        assert (
            2 <= starting_number <= max_number
        ), "Starting number must be between 2 and max_number."

        self.starting_number = starting_number
        self.max_number = max_number

        # Define action and observation space
        # Actions are integers from 0 to max_number inclusive
        self.action_space = spaces.Discrete(self.max_number + 1)
        self.observation_space = spaces.Box(
            low=1, high=self.max_number, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Return observation and info

    def step(self, action):
        # Check if the move is valid
        if (
            action <= 1
            or action >= self.current_number
            or self.current_number % action != 0
        ):
            # Invalid move
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move; subtract the chosen proper divisor from the current number
        self.current_number -= action

        # Check if the game is over
        if self.current_number <= 1 or not self.has_proper_divisors(
            self.current_number
        ):
            # Current player wins
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Continue the game; switch players
        self.current_player *= -1  # Switch player
        return (
            np.array([self.current_number], dtype=np.int32),
            0,
            False,
            False,
            {},
        )

    def render(self):
        return f"Current Number: {self.current_number}"

    def valid_moves(self):
        return [
            d for d in range(2, self.current_number) if self.current_number % d == 0
        ]

    def has_proper_divisors(self, number):
        for d in range(2, number):
            if number % d == 0:
                return True
        return False
