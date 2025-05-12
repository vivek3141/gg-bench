import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum value for N
        self.max_N = 1000

        # Define action and observation space
        # Actions correspond to integers from 0 to max_N (inclusive)
        self.action_space = spaces.Discrete(self.max_N + 1)

        # Observation is N, an integer between 1 and max_N
        self.observation_space = spaces.Box(
            low=1, high=self.max_N, shape=(1,), dtype=np.int32
        )

        # Initialize game state
        self.N = None
        self.current_player = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Starting number N (>1), here we start with 30 as in the example
        self.N = 30
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Observation, info

    def step(self, action):
        # If the game is already over, return the current state
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        # Validate the action
        if action < 2 or action >= self.N:
            # Invalid move: action must be between 2 and N-1
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        if self.N % action != 0:
            # Invalid move: action must be a proper divisor of N
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Valid move: subtract the chosen divisor from N
        self.N -= action

        # Check if the game has been won
        if self.N <= 1 or len(self.get_proper_divisors(self.N)) == 0:
            # Opponent cannot make a move; current player wins
            self.done = True
            reward = 1
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Continue the game, switch to the next player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        reward = 0
        return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def render(self):
        return f"Current N: {self.N}, Current Player: {self.current_player}"

    def valid_moves(self):
        # Return a list of valid moves (proper divisors of N excluding 1 and N)
        return self.get_proper_divisors(self.N)

    @staticmethod
    def get_proper_divisors(n):
        # Calculate proper divisors of n excluding 1 and n itself
        return [i for i in range(2, n) if n % i == 0]
