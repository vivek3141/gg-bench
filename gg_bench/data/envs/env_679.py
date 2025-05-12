import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=30):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: actions from 0 to 99, representing possible divisors from 1 to 100
        self.action_space = spaces.Discrete(100)

        # Observation space: the current number
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([1000]), shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.starting_number = starting_number
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and "starting_number" in options:
            self.current_number = options["starting_number"]
        else:
            self.current_number = self.starting_number

        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        # Map action to divisor
        divisor = (
            action + 1
        )  # Actions from 0 to 99 correspond to divisors from 1 to 100

        valid_divisors = self._get_proper_divisors(self.current_number)

        if self.done:
            reward = 0
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        if divisor not in valid_divisors:
            # Invalid move
            reward = -10
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        # Subtract the chosen divisor from current number
        self.current_number -= divisor

        # Check if opponent has valid moves
        opponent_valid_divisors = self._get_proper_divisors(self.current_number)

        if len(opponent_valid_divisors) == 0 or self.current_number == 1:
            # Opponent cannot move, current player wins
            reward = 1
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to next player
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0
        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        # Return a string representation of the current state
        return f"Current Number: {self.current_number}, Player {self.current_player}'s turn"

    def valid_moves(self):
        # Return list of valid actions (indices)
        valid_divisors = self._get_proper_divisors(self.current_number)
        valid_actions = [
            d - 1 for d in valid_divisors
        ]  # Convert divisors to action indices
        return valid_actions

    def _get_proper_divisors(self, n):
        # Return list of proper divisors of n (excluding 1 and n)
        if n <= 3:
            return []  # No proper divisors (excluding 1 and n)
        divisors = set()
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                divisors.add(i)
                if i != n // i and n // i != n:
                    divisors.add(n // i)
        divisors.discard(n)  # Ensure n itself is excluded
        return sorted(divisors)
