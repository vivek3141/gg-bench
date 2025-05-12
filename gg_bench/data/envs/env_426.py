import numpy as np
import gymnasium as gym
from gymnasium import spaces


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True  # 2 and 3 are primes
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


class CustomEnv(gym.Env):
    def __init__(self, starting_number=100):
        super(CustomEnv, self).__init__()

        self.starting_number = starting_number
        self.current_number = starting_number
        self.current_player = 1  # Player 1 starts

        # Define action and observation space
        # Actions correspond to the number to subtract (the proper divisor)
        # Since we cannot subtract a number greater than starting_number
        # Action space ranges from 0 to starting_number
        self.action_space = spaces.Discrete(self.starting_number + 1)

        # Observation space is the current number
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([self.starting_number]), dtype=np.int32
        )

        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Validate action
        if (
            action <= 0
            or action >= self.current_number
            or self.current_number % action != 0
        ):
            # Invalid action
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Valid action
        self.current_number -= action

        # Check for game over conditions
        if self.current_number == 1:
            # Player who made the move loses
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -1,
                True,
                False,
                {},
            )
        elif is_prime(self.current_number):
            # Player who made the move wins
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )
        else:
            # Check if next player cannot make a valid move
            if len(self.valid_moves()) == 0:
                # Next player loses
                self.done = True
                # The current player wins
                return (
                    np.array([self.current_number], dtype=np.int32),
                    1,
                    True,
                    False,
                    {},
                )

            # Switch player
            self.current_player = 3 - self.current_player
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        player = "Player 1" if self.current_player == 1 else "Player 2"
        return f"Current number: {self.current_number}, {player}'s turn"

    def valid_moves(self):
        # Proper divisors of current_number are numbers less than current_number that divide it evenly
        return [
            action
            for action in range(1, self.current_number)
            if self.current_number % action == 0
        ]
