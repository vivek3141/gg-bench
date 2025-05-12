import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the starting number
        self.starting_number = 60

        # Define action and observation space
        # Action space: Possible factors from 2 up to starting_number
        self.action_space = spaces.Discrete(self.starting_number + 1)

        # Observation space: The current shared number
        self.observation_space = spaces.Box(
            low=1, high=np.inf, shape=(1,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Set the shared number to the starting number
        self.shared_number = self.starting_number

        # Set the current player (1 or -1)
        self.current_player = 1

        # Game termination flag
        self.done = False

        # Return the initial observation and info dict
        return np.array([self.shared_number], dtype=np.int32), {}

    def step(self, action):
        # Check if the game has already ended
        if self.done:
            return (
                np.array([self.shared_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Get the list of valid moves
        valid_actions = self.valid_moves()

        # Check if the action is valid
        if action not in valid_actions:
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Perform the action
        self.shared_number = self.shared_number // action

        # Check for victory conditions
        if self.shared_number == 1:
            # Current player wins
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        elif self.is_prime(self.shared_number):
            # Next player cannot move; current player wins
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        else:
            # Switch to the next player
            self.current_player *= -1
            return (
                np.array([self.shared_number], dtype=np.int32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        # Provide a textual representation of the game state
        return (
            f"Current Shared Number: {self.shared_number}\n"
            f"Player's Turn: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
            f"Available Factors: {self.valid_moves()}\n"
        )

    def valid_moves(self):
        # Calculate valid factors of the current shared number
        factors = [
            i for i in range(2, self.shared_number) if self.shared_number % i == 0
        ]
        return factors

    def is_prime(self, n):
        # Check if a number is prime
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
