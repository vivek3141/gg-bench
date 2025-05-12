import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=100):
        super(CustomEnv, self).__init__()

        self.starting_number = starting_number
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Define action and observation space
        # Actions are integers from 0 to starting_number
        self.action_space = spaces.Discrete(self.starting_number + 1)
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([self.starting_number]), dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        self.current_number = int(self.current_number / action)

        if self.current_number == 1:
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check if next player has valid moves
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            # Next player cannot make a valid move, current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Game continues
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0
        return np.array([self.current_number], dtype=np.int32), reward, False, False, {}

    def render(self):
        return f"Player {self.current_player}'s turn. Current Number: {self.current_number}"

    def valid_moves(self):
        N = self.current_number
        return [i for i in range(2, N) if N % i == 0]
