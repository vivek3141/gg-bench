import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.max_N = 100  # Maximum allowed N value
        self.action_space = spaces.Discrete(
            self.max_N
        )  # Actions are integers from 0 to max_N - 1

        # Observation space contains the current N value
        self.observation_space = spaces.Box(
            low=1, high=self.max_N, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = random.randint(10, 30)  # Starting N between 10 and 30 for variability
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def get_proper_divisors(self, n):
        # Proper divisors are all divisors of n excluding 1 and n itself
        return [i for i in range(2, n) if n % i == 0]

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        valid_divisors = self.get_proper_divisors(self.N)

        # Check if the action is valid
        if action not in valid_divisors or self.N - action <= 0:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Valid move
        self.N -= action

        # Check if the next player can make a move
        valid_next_divisors = self.get_proper_divisors(self.N)
        if len(valid_next_divisors) == 0:
            # Opponent cannot make a move; current player wins
            self.done = True
            reward = 1  # Reward for winning
            return np.array([self.N], dtype=np.int32), reward, True, False, {}
        else:
            # Switch player and continue the game
            self.current_player *= -1
            reward = 0  # No reward for a regular valid move
            return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def render(self):
        # Return a string representation of the current game state
        player_str = "Player 1" if self.current_player == 1 else "Player 2"
        return f"Current N: {self.N}, Current Player: {player_str}"

    def valid_moves(self):
        # Return a list of valid actions for the current state
        return self.get_proper_divisors(self.N)
