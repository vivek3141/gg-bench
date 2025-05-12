import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, initial_N=16):
        super(CustomEnv, self).__init__()

        self.initial_N = initial_N
        self.N = initial_N

        self.MAX_N = initial_N  # Maximum possible value of N during the game

        # Define action and observation spaces
        # Actions correspond to possible divisors from 0 to MAX_N (inclusive)
        self.action_space = spaces.Discrete(self.MAX_N + 1)

        # Observation is the current value of N
        self.observation_space = spaces.Box(
            low=1, high=self.MAX_N, shape=(1,), dtype=np.int32
        )

        self.current_player = 1  # Player 1 starts the game
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.initial_N
        self.current_player = 1
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Observation, info

    def step(self, action):
        if self.done:
            return (
                np.array([self.N], dtype=np.int32),
                0.0,
                True,
                False,
                {},
            )  # Game is over

        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid move
            self.done = True
            return np.array([self.N], dtype=np.int32), -10.0, True, False, {}

        # Apply the action
        self.N = self.N // action

        # Check if the opponent has any valid moves
        opponent_valid_moves = self.valid_moves()
        if len(opponent_valid_moves) == 0:
            # Current player wins
            self.done = True
            return np.array([self.N], dtype=np.int32), 1.0, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        return np.array([self.N], dtype=np.int32), -10.0, False, False, {}

    def render(self):
        return f"Current N: {self.N}, Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        # Find all proper divisors of N (excluding 1 and N itself)
        proper_divisors = [d for d in range(2, self.N) if self.N % d == 0]
        return proper_divisors
