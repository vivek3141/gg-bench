import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: integers from 0 to 20 (inclusive)
        self.action_space = spaces.Discrete(21)
        # Observation space: the current counter value (from 0 to 20)
        self.observation_space = spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = 20
        self.current_player = 1  # You can use 1 and -1 to represent the two players
        self.done = False
        return np.array([self.counter], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            return (
                np.array([self.counter], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move
        self.counter -= action

        if self.counter == 0:
            # Current player wins by reducing counter to zero
            self.done = True
            return (
                np.array([self.counter], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Temporarily switch player to check if next player has valid moves
        self.current_player *= -1
        next_player_moves = self.valid_moves()

        if len(next_player_moves) == 0:
            # Next player cannot move; current player wins
            self.current_player *= -1  # Switch back to current player
            self.done = True
            return (
                np.array([self.counter], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Game continues; current player remains switched
        return (
            np.array([self.counter], dtype=np.int32),
            0,
            False,
            False,
            {},
        )

    def render(self):
        return f"Current counter: {self.counter}\nCurrent player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        # Calculate valid moves based on the current counter
        if self.counter == 1:
            # No valid moves if counter is 1
            return []
        elif self.is_prime(self.counter):
            # Counter is prime; can subtract 1
            return [1]
        else:
            # List proper divisors excluding 1 and the counter itself
            divisors = [
                i
                for i in range(2, self.counter)
                if self.counter % i == 0 and i != self.counter
            ]
            return divisors

    def is_prime(self, n):
        # Helper method to check if a number is prime
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
