import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the maximum starting number
        self.N_max = 100  # You can adjust this value as needed

        # Define action and observation space
        # Actions are integers from 0 to N_max inclusive
        self.action_space = spaces.Discrete(self.N_max + 1)

        # Observation is the current N, a float32 number between 1 and N_max
        self.observation_space = spaces.Box(
            low=np.array([1.0]),
            high=np.array([self.N_max]),
            shape=(1,),
            dtype=np.float32,
        )

        # Initialize variables
        self.current_N = None
        self.starting_N = self.N_max  # Starting number can be adjusted
        self.current_player = None
        self.done = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_N = self.starting_N
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_N], dtype=np.float32), {}  # Observation, info

    def step(self, action):
        if self.done:
            # If the game is over, return current observation with zero rewards
            return (
                np.array([self.current_N], dtype=np.float32),
                0,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        D = action

        # Check if action D is a valid proper divisor of current_N
        if D <= 1 or D >= self.current_N or self.current_N % D != 0:
            # Invalid move
            self.done = True
            reward = -10
            return (
                np.array([self.current_N], dtype=np.float32),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        self.current_N = self.current_N // D

        # Check for win conditions
        if self.current_N == 1:
            # Current player wins by reducing N to 1
            self.done = True
            reward = 1
            return (
                np.array([self.current_N], dtype=np.float32),
                reward,
                True,
                False,
                {},
            )
        elif self.is_prime(self.current_N):
            # Next player cannot make a move; current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_N], dtype=np.float32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Game continues; switch player
            self.current_player = 3 - self.current_player  # Switches between 1 and 2
            reward = 0
            return (
                np.array([self.current_N], dtype=np.float32),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        return f"Current N is {self.current_N}."

    def valid_moves(self):
        return [D for D in range(2, int(self.current_N)) if self.current_N % D == 0]

    @staticmethod
    def is_prime(n):
        if n <= 1:
            return False
        elif n <= 3:
            return True
        elif n % 2 == 0:
            return False
        sqrt_n = int(np.sqrt(n)) + 1
        for i in range(3, sqrt_n, 2):
            if n % i == 0:
                return False
        return True
