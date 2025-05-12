import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Set starting N
        self.starting_N = 30
        self.N = self.starting_N

        # Define action and observation space
        # Actions from 0 to 28 correspond to factors from 2 to 30
        self.action_space = spaces.Discrete(29)

        # Observation space: current N
        self.observation_space = spaces.Box(
            low=np.array([2]), high=np.array([30]), shape=(1,), dtype=np.float32
        )

        self.current_player = 1
        self.done = False

    def reset(self, seed=None, options=None):
        # Initialize starting number
        super().reset(seed=seed)
        self.N = self.starting_N
        self.current_player = 1
        self.done = False
        return np.array([self.N], dtype=np.float32), {}

    def step(self, action):
        if self.done:
            # If the game is already over, return current observation
            return np.array([self.N], dtype=np.float32), 0, True, False, {}

        # Map action to factor (actions 0-28 correspond to factors 2-30)
        factor = action + 2

        # Get valid factors
        valid_factors = [i for i in range(2, self.N) if self.N % i == 0]

        if factor not in valid_factors:
            # Invalid move
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.float32), reward, True, False, {}

        # Valid move
        # Update N
        self.N = self.N // factor

        # Check if N is prime
        if self.is_prime(self.N):
            # Current player wins
            self.done = True
            reward = 1
            return np.array([self.N], dtype=np.float32), reward, True, False, {}
        else:
            # Game continues
            # Switch player
            self.current_player *= -1
            return np.array([self.N], dtype=np.float32), 0, False, False, {}

    def render(self):
        # Get valid factors for display
        valid_factors = [action + 2 for action in self.valid_moves()]
        return f"Current N: {self.N}, Player: {self.current_player}, Available factors: {valid_factors}"

    def valid_moves(self):
        # Get valid factors
        valid_factors = [i for i in range(2, self.N) if self.N % i == 0]
        # Map factors to actions
        valid_actions = [factor - 2 for factor in valid_factors]
        return valid_actions

    def is_prime(self, n):
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
