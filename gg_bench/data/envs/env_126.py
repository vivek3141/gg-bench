import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: integers from 0 to 60 (possible factors)
        self.action_space = spaces.Discrete(61)  # Actions are integers from 0 to 60

        # Observation space: current shared number (from 1 to 60)
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([60]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = 60  # Starting number
        self.current_player = 1
        self.done = False
        return (
            np.array([self.shared_number], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            return np.array([self.shared_number], dtype=np.int32), 0, True, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move, player loses
            self.done = True
            reward = -10
            return (
                np.array([self.shared_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        self.shared_number = self.shared_number // action

        # Check if the shared number is 1
        if self.shared_number == 1:
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.shared_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Check if the next player has any valid moves
            next_valid_moves = self.get_proper_factors(self.shared_number)
            if not next_valid_moves:
                # Next player cannot move, current player wins
                self.done = True
                reward = 1
                return (
                    np.array([self.shared_number], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )
            else:
                # Switch current player
                self.current_player = 3 - self.current_player
                reward = 0
                return (
                    np.array([self.shared_number], dtype=np.int32),
                    reward,
                    False,
                    False,
                    {},
                )

    def render(self):
        # Return a visual representation of the environment state as a string
        info_str = f"Current shared number: {self.shared_number}\n"
        info_str += f"Current player: Player {self.current_player}\n"
        info_str += f"Valid moves: {self.valid_moves()}\n"
        return info_str

    def valid_moves(self):
        if self.done:
            return []

        return self.get_proper_factors(self.shared_number)

    def get_proper_factors(self, n):
        factors = []
        for i in range(2, n):
            if n % i == 0:
                factors.append(i)
        return factors
