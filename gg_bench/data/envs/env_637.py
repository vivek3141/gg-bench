import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=60):
        super(CustomEnv, self).__init__()

        self.starting_number = starting_number

        # Define action and observation space
        # Actions are possible divisors from 0 to starting_number inclusive
        self.action_space = spaces.Discrete(self.starting_number + 1)
        # Observation is the current number
        self.observation_space = spaces.Box(
            low=1, high=self.starting_number, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.done = False
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return (np.array([self.current_number], dtype=np.int32), 0, True, False, {})

        valid_divisors = self.valid_moves()
        if action not in valid_divisors:
            # Invalid move
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Valid move
        self.current_number = self.current_number // action

        # Check if next player has any valid moves
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            # Current player wins
            self.done = True
            return (np.array([self.current_number], dtype=np.int32), 1, True, False, {})
        else:
            # Game continues
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        # Return a string representation of the current state
        return f"Current number: {self.current_number}"

    def valid_moves(self):
        # Return a list of valid actions (proper divisors of the current number)
        divisors = []
        for i in range(2, self.current_number):
            if self.current_number % i == 0:
                divisors.append(i)
        return divisors

    def close(self):
        pass
