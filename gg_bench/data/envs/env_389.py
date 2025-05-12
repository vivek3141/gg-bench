import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action space: indices 0,1,2,3 correspond to subtracting primes 2,3,5,7
        self.action_space = spaces.Discrete(4)
        # Observation space: the shared number
        self.observation_space = spaces.Box(low=0, high=50, shape=(1,), dtype=np.int32)

        self.allowed_primes = [2, 3, 5, 7]
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = 50
        self.current_player = 1
        self.done = False
        return np.array([self.shared_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.shared_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Check if current player has any valid moves
        current_valid_moves = [
            i for i, p in enumerate(self.allowed_primes) if self.shared_number - p >= 0
        ]
        if not current_valid_moves:
            # Current player cannot move, they lose the game
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                -1,
                True,
                False,
                {},
            )

        # Check if action is among valid moves
        if action not in current_valid_moves:
            # Invalid move
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Valid move
        prime = self.allowed_primes[action]
        self.shared_number -= prime

        # Check for win
        if self.shared_number == 0:
            # Current player wins
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Check if opponent has any valid moves
        opponent_valid_moves = [
            i for i, p in enumerate(self.allowed_primes) if self.shared_number - p >= 0
        ]
        if not opponent_valid_moves:
            # Opponent cannot move, current player wins
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        return (
            np.array([self.shared_number], dtype=np.int32),
            0,
            False,
            False,
            {},
        )

    def render(self):
        return (
            f"Player {self.current_player}'s Turn\nShared Number: {self.shared_number}"
        )

    def valid_moves(self):
        return [
            i for i, p in enumerate(self.allowed_primes) if self.shared_number - p >= 0
        ]
