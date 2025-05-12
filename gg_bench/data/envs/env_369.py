import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N_start=12, N_max=100):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.N_max = N_max
        self.N_start = N_start
        self.action_space = spaces.Discrete(self.N_max + 1)  # Actions from 0 to N_max

        # Observation space is [N, current_player], N ranges from 2 to N_max, current_player is -1 or 1
        self.observation_space = spaces.Box(
            low=np.array([2, -1], dtype=np.int32),
            high=np.array([self.N_max, 1], dtype=np.int32),
            shape=(2,),
            dtype=np.int32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_start  # Starting number
        self.current_player = 1  # Player 1 starts (1 or -1)
        self.done = False
        return (
            np.array([self.N, self.current_player], dtype=np.int32),
            {},
        )  # observation, info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return (
                np.array([self.N, self.current_player], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Check if action is invalid: not a proper divisor or outside valid range
        if action <= 1 or action >= self.N or self.N % action != 0:
            reward = -10  # Invalid move
            self.done = True
            return (
                np.array([self.N, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        else:
            self.N -= action
            # Check if the game is over (no valid moves for next player)
            if self.N <= 1 or len(self.valid_moves()) == 0:
                reward = 1  # Current player wins
                self.done = True
                return (
                    np.array([self.N, self.current_player], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )
            else:
                # Switch player
                self.current_player *= -1
                reward = 0
                return (
                    np.array([self.N, self.current_player], dtype=np.int32),
                    reward,
                    False,
                    False,
                    {},
                )

    def render(self):
        return f"Current N: {self.N}, Current Player: {self.current_player}"

    def valid_moves(self):
        # Return list of valid proper divisors of N (excluding 1 and N)
        return [i for i in range(2, self.N) if self.N % i == 0]
