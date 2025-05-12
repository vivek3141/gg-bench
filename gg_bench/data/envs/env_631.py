import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_total=100):
        super(CustomEnv, self).__init__()

        # Primes less than 20
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19]
        self.action_space = spaces.Discrete(len(self.primes))

        # Observation space: current total
        self.starting_total = starting_total
        self.current_total = self.starting_total
        self.observation_space = spaces.Box(
            low=np.array([0]),
            high=np.array([self.starting_total]),
            shape=(1,),
            dtype=np.int32,
        )

        self.current_player = 1  # Player 1 starts
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_total = self.starting_total
        self.current_player = 1  # Reset to Player 1
        self.done = False
        return (
            np.array([self.current_total], dtype=np.int32),
            {},
        )  # Return observation and info

    def step(self, action):
        if self.done:
            # Game has already ended
            return np.array([self.current_total], dtype=np.int32), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalize invalid move
            return (
                np.array([self.current_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        prime = self.primes[action]
        self.current_total -= prime

        if self.current_total == 0:
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Check if next player has valid moves
            opponent_valid_actions = self.valid_moves()
            if not opponent_valid_actions:
                # Opponent cannot make a valid move, current player wins
                self.done = True
                reward = 1
                return (
                    np.array([self.current_total], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )
            else:
                # Switch to next player
                self.current_player *= -1
                reward = 0
                return (
                    np.array([self.current_total], dtype=np.int32),
                    reward,
                    False,
                    False,
                    {},
                )

    def render(self):
        return f"Current Total: {self.current_total}\n"

    def valid_moves(self):
        # Return a list of valid action indices based on the current total
        return [
            i for i in range(len(self.primes)) if self.primes[i] <= self.current_total
        ]
