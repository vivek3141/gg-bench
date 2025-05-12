import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.N_MAX = 100  # Maximum value for N
        self.action_space = spaces.Discrete(
            self.N_MAX - 1
        )  # Actions correspond to divisors D from 2 to N_MAX
        self.observation_space = spaces.Box(
            low=1, high=self.N_MAX, shape=(1,), dtype=np.int32
        )
        self.N = None  # Current number
        self.done = False  # Game over flag

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Set the starting number N
        if options and "starting_number" in options:
            self.N = options["starting_number"]
        else:
            self.N = 16  # Default starting number
        self.done = False
        observation = np.array([self.N], dtype=np.int32)
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            observation = np.array([self.N], dtype=np.int32)
            return observation, 0, True, False, {}
        # Get the valid moves
        valid_actions = self.valid_moves()
        # Check if there are any valid moves
        if not valid_actions:
            # No valid moves available, player loses
            self.done = True
            reward = -10
            observation = np.array([self.N], dtype=np.int32)
            return observation, reward, True, False, {}
        # Check if the action is valid
        if action not in valid_actions:
            # Invalid move, player loses
            self.done = True
            reward = -10
            observation = np.array([self.N], dtype=np.int32)
            return observation, reward, True, False, {}
        # Perform the valid action
        D = action + 2  # Map action index to divisor D
        self.N = self.N // D
        observation = np.array([self.N], dtype=np.int32)
        # Check if opponent has any valid moves
        opponent_valid_actions = self.valid_moves()
        if not opponent_valid_actions:
            # Opponent cannot move, current player wins
            self.done = True
            reward = 1
            return observation, reward, True, False, {}
        else:
            # Game continues
            reward = 0
            return observation, reward, False, False, {}

    def render(self):
        return f"Current number: {self.N}"

    def valid_moves(self):
        # Get the list of valid moves as action indices
        moves = []
        for D in range(2, min(self.N + 1, self.N_MAX + 1)):
            if D < self.N and self.N % D == 0:
                action = D - 2  # Map divisor D to action index
                moves.append(action)
        return moves
