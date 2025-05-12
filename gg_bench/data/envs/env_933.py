import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: Multipliers from 2 to 9 (indices 0 to 7)
        self.action_space = spaces.Discrete(8)

        # Define observation space: Current number (scalar)
        # Since the number can grow large, we use a large upper bound
        self.observation_space = spaces.Box(
            low=1, high=np.iinfo(np.int64).max, shape=(1,), dtype=np.int64
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array([self.current_number], dtype=np.int64)
        return observation, {}  # Return observation and info

    def step(self, action):
        # Map action index to multiplier (2 to 9)
        multiplier = (
            action + 2
        )  # Action 0 -> Multiplier 2, ..., Action 7 -> Multiplier 9

        # Validate action
        if multiplier < 2 or multiplier > 9 or self.done:
            # Invalid action or game already over
            observation = np.array([self.current_number], dtype=np.int64)
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Current player's move
        self.current_number *= multiplier

        # Check if current number is prime
        if self.is_prime(self.current_number):
            # Current player loses
            self.done = True
            observation = np.array([self.current_number], dtype=np.int64)
            return observation, -10, True, False, {}  # Player loses

        # Opponent's move
        opponent_multiplier = self.opponent_policy()
        self.current_number *= opponent_multiplier

        # Check if current number is prime after opponent's move
        if self.is_prime(self.current_number):
            # Opponent loses, current player wins
            self.done = True
            observation = np.array([self.current_number], dtype=np.int64)
            return observation, 1, True, False, {}  # Player wins

        # Game continues, switch turns
        self.current_player = 1  # It's always the agent's turn in this setup
        observation = np.array([self.current_number], dtype=np.int64)
        return observation, -10, False, False, {}  # Valid move, game continues

    def render(self):
        # Provide a string representation of the current state
        return f"Current Number: {self.current_number}"

    def valid_moves(self):
        # All multipliers from 2 to 9 are always valid
        return list(range(8))  # Actions 0 to 7

    def opponent_policy(self):
        # Opponent selects a random valid multiplier
        return random.randint(2, 9)

    def is_prime(self, n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        w = 2
        while i * i <= n and i < 1e6:  # Limit to avoid performance issues
            if n % i == 0:
                return False
            i += w
            w = 6 - w
        return True
