import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Primes allowed for division
        self.primes = [2, 3, 5, 7]
        self.num_primes = len(self.primes)

        # Define action and observation space
        # Actions correspond to indices of the primes list
        self.action_space = spaces.Discrete(self.num_primes)

        # Observation is the current value of N
        self.initial_N = 210  # Initial value of N
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([self.initial_N]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.initial_N
        self.current_player = 1  # Player 1 starts
        self.done = False
        info = {}
        return np.array([self.N], dtype=np.int32), info  # Observation and info

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0.0, True, False, {}

        selected_prime = self.primes[action]
        if self.N % selected_prime != 0:
            # Invalid move
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int32), reward, True, False, {}
        else:
            # Valid move
            self.N = self.N // selected_prime

            if self.N == 1:
                # Current player wins
                self.done = True
                reward = 1
                return np.array([self.N], dtype=np.int32), reward, True, False, {}
            else:
                # Switch player
                self.current_player *= -1
                reward = 0
                return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def render(self):
        return f"Current N is {self.N}"

    def valid_moves(self):
        valid_moves = []
        for idx, prime in enumerate(self.primes):
            if self.N % prime == 0:
                valid_moves.append(idx)
        return valid_moves
