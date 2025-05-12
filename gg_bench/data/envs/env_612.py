import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Allowed prime divisors
        self.allowed_divisors = [2, 3, 5, 7]

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.allowed_divisors))
        self.observation_space = spaces.Box(low=1, high=1e9, shape=(1,), dtype=np.int32)

        # Set initial game state
        self.initial_number = 1000  # Default starting number
        self.current_number = self.initial_number
        self.current_player = 1  # Player 1 starts
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.initial_number
        self.current_player = 1
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        valid_actions = self.valid_moves()

        if not valid_actions:
            # Current player cannot make a valid move and loses
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        if action not in valid_actions:
            # Invalid action selected; current player loses
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
        divisor = self.allowed_divisors[action]
        self.current_number = self.current_number // divisor

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

        # Switch players
        self.current_player = 2 if self.current_player == 1 else 1
        reward = -10  # Penalty for valid move without winning
        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        return f"Current Number: {self.current_number}"

    def valid_moves(self):
        # Returns list of valid action indices
        valid_actions = []
        for idx, divisor in enumerate(self.allowed_divisors):
            if self.current_number % divisor == 0:
                valid_actions.append(idx)
        return valid_actions
