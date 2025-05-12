import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N_starting=50):
        super(CustomEnv, self).__init__()

        # Starting Number
        self.N_starting = N_starting

        # Define action and observation space
        # Actions are integers from 1 to N_starting inclusive
        self.action_space = spaces.Discrete(
            self.N_starting + 1
        )  # Actions: 0 to N_starting
        self.observation_space = spaces.Box(
            low=1, high=self.N_starting, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_starting
        self.done = False
        self.current_player = 1  # Player 1 starts
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return (
                self._get_obs(),
                0,
                True,
                False,
                {},
            )  # No reward if game is already done

        # Check if N is 1 at the start of the turn
        if self.N == 1:
            # Current player loses
            reward = -1
            self.done = True
            return self._get_obs(), reward, True, False, {}

        valid_actions = self.valid_moves()

        # Check if action is valid
        if action not in valid_actions:
            # Invalid move
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}

        # Subtract the chosen divisor from N
        self.N -= action

        # Check if N is 1 after the move
        if self.N == 1:
            # Current player wins
            reward = 1
            self.done = True
            return self._get_obs(), reward, True, False, {}

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0
        return self._get_obs(), reward, False, False, {}

    def render(self):
        return f"Current N: {self.N}, Current Player: Player {self.current_player}"

    def valid_moves(self):
        return self._get_proper_divisors(self.N)

    def _get_obs(self):
        return np.array([self.N], dtype=np.int32)

    @staticmethod
    def _get_proper_divisors(N):
        return [i for i in range(1, N) if N % i == 0]
