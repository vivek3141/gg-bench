import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.starting_N = 60  # Default starting number

        # Define action and observation space
        # The action_space is Discrete(starting_N + 1), actions are integers from 0 to starting_N
        self.action_space = spaces.Discrete(self.starting_N + 1)

        # The observation_space is the current N
        self.observation_space = spaces.Box(
            low=0, high=self.starting_N, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.starting_N
        self.current_player = 1  # 1 or -1 to represent players
        self.done = False
        observation = np.array([self.N], dtype=np.int32)
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            reward = -10
            terminated = True
            truncated = False
            observation = np.array([self.N], dtype=np.int32)
            return observation, reward, terminated, truncated, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            terminated = True
            truncated = False
            observation = np.array([self.N], dtype=np.int32)
            return observation, reward, terminated, truncated, {}

        # Valid move
        self.N = self.N - action
        observation = np.array([self.N], dtype=np.int32)

        # Check if next player has any valid moves
        next_valid_moves = self.get_proper_divisors(self.N)
        if len(next_valid_moves) == 0:
            # Next player cannot move, current player wins
            self.done = True
            reward = 1
            terminated = True
            truncated = False
            return observation, reward, terminated, truncated, {}
        else:
            # Game continues
            self.current_player *= -1
            self.done = False
            reward = 0
            terminated = False
            truncated = False
            return observation, reward, terminated, truncated, {}

    def render(self):
        return f"Current Number (N): {self.N}"

    def valid_moves(self):
        # Returns the list of valid actions
        # Valid actions are proper divisors of current N
        return self.get_proper_divisors(self.N)

    def get_proper_divisors(self, n):
        divisors = []
        for i in range(2, n):
            if n % i == 0:
                divisors.append(i)
        return divisors
