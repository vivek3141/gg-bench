import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space allows integers from 0 to 99
        self.action_space = spaces.Discrete(100)
        # The observation is the current number N
        self.observation_space = spaces.Box(
            low=np.array([2]), high=np.array([1000]), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = 30  # Starting number can be set as desired
        self.done = False
        self.current_player = 1  # For internal management
        return np.array([self.N], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        if self.done:
            # If the game is already over
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        # Check if the current player has any valid moves
        valid_moves = self.valid_moves()
        if not valid_moves:
            # Current player cannot make a move and loses
            self.done = True
            reward = -1  # Current player loses
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Check if action is a valid move
        if action not in valid_moves:
            # Invalid move made
            self.done = True
            reward = -10  # Penalty for invalid move
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Apply the action
        self.N -= action  # Subtract the chosen proper divisor from N

        # Check if the next player has any valid moves
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            # Next player cannot move; current player wins
            self.done = True
            reward = +1  # Current player wins
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Game continues; switch to the next player
        self.current_player *= -1  # Switch player (for internal use)
        return np.array([self.N], dtype=np.int32), 0, False, False, {}

    def render(self):
        # Return a string representation of the current state
        return f"Current Number (N): {self.N}"

    def valid_moves(self):
        # Calculate proper divisors of N (excluding 1 and N)
        proper_divisors = []
        for i in range(2, self.N):
            if self.N % i == 0:
                proper_divisors.append(i)
        return proper_divisors
