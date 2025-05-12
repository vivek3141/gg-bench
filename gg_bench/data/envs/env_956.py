import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, initial_N=30):
        super(CustomEnv, self).__init__()

        self.initial_N = initial_N

        # Define action and observation spaces
        # Action space: actions are integers from 0 to initial_N - 1, mapping to divisors from 1 to initial_N
        self.action_space = spaces.Discrete(self.initial_N)
        # Observation space: current N, integer from 0 to initial_N
        self.observation_space = spaces.Box(
            low=np.array([0]), high=np.array([self.initial_N]), dtype=np.int32
        )

        # Initialize state
        self.current_N = None
        self.current_player = None
        self.done = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_N = self.initial_N
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array([self.current_N], dtype=np.int32)
        info = {}
        return observation, info

    def step(self, action):
        info = {}

        if self.done:
            observation = np.array([self.current_N], dtype=np.int32)
            return observation, 0, True, False, info

        # Map action index to divisor
        divisor = action + 1

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            self.done = True
            terminated = True
            truncated = False
            observation = np.array([self.current_N], dtype=np.int32)
            return observation, reward, terminated, truncated, info

        # Valid move
        self.current_N -= divisor
        observation = np.array([self.current_N], dtype=np.int32)

        if self.current_N == 0:
            # Current player wins
            reward = 1
            self.done = True
            terminated = True
            truncated = False
            return observation, reward, terminated, truncated, info

        elif self.current_N < 0:
            # Should not happen if action is valid
            reward = -10
            self.done = True
            terminated = True
            truncated = False
            return observation, reward, terminated, truncated, info

        else:
            # Game continues
            self.current_player *= -1  # Switch player
            reward = 0
            terminated = False
            truncated = False
            return observation, reward, terminated, truncated, info

    def render(self):
        return f"Current N: {self.current_N}, Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        N = self.current_N
        if N <= 0:
            return []
        divisors = [i for i in range(1, N + 1) if N % i == 0]
        # Map divisors to action indices
        valid_actions = [d - 1 for d in divisors]
        return valid_actions
