import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Allowed prime numbers for subtraction
        self.primes = [2, 3, 5, 7, 11, 13]

        # Action space: Select a prime number to subtract (indices 0 to 5)
        self.action_space = spaces.Discrete(len(self.primes))

        # Observation space: The current total (from 0 to 100)
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.total = 100  # Starting total
        self.current_player = 1  # Player 1: 1, Player 2: -1
        self.done = False
        return np.array([self.total], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        if self.done:
            return np.array([self.total], dtype=np.int32), 0, True, False, {}

        # Check if action is a valid index in action space
        if not self.action_space.contains(action):
            # Invalid action (invalid index)
            reward = -10
            self.current_player *= -1  # Switch player
            return np.array([self.total], dtype=np.int32), reward, False, False, {}

        prime = self.primes[action]

        # Check if subtracting the prime results in negative total
        if self.total - prime < 0:
            # Invalid move (would result in negative total)
            reward = -10
            self.current_player *= -1  # Switch player
            return np.array([self.total], dtype=np.int32), reward, False, False, {}

        # Valid move
        self.total -= prime

        if self.total == 0:
            # Current player wins
            reward = 1
            self.done = True
            return np.array([self.total], dtype=np.int32), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            self.current_player *= -1  # Switch player
            return np.array([self.total], dtype=np.int32), reward, False, False, {}

    def render(self):
        return f"Current total: {self.total}, Current player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        # Return list of valid action indices
        valid_actions = []
        for i, prime in enumerate(self.primes):
            if self.total - prime >= 0:
                valid_actions.append(i)
        return valid_actions
