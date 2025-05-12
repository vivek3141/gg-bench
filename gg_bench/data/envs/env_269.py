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

        # The action space consists of all possible divisors from 2 up to the starting number
        self.action_space = spaces.Discrete(
            self.starting_number + 1
        )  # Actions from 0 to starting_number

        # Observation space is the current number
        self.observation_space = spaces.Box(
            low=1, high=self.starting_number, shape=(1,), dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Return observation and info

    def step(self, action):
        action = int(action)

        # Invalid move if action is not a valid divisor
        if not self._is_valid_action(action):
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                self.done,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Update the current number
        self.current_number = self.current_number // action

        # Check if the next player has any valid moves
        if not self._has_valid_moves():
            self.done = True
            reward = 1  # Current player wins
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                self.done,
                False,
                {},
            )

        # Switch to the next player
        self.current_player *= -1
        reward = 0
        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        return f"Current Number: {self.current_number}, Player: {'1' if self.current_player == 1 else '2'}"

    def valid_moves(self):
        # Return a list of valid actions (divisors) as indices of the action space
        return [
            action
            for action in range(2, self.current_number)
            if self.current_number % action == 0
        ]

    def _is_valid_action(self, action):
        if action < 2 or action >= self.current_number:
            return False
        if self.current_number % action != 0:
            return False
        return True

    def _has_valid_moves(self):
        # Check if there are any valid divisors for the current number
        for divisor in range(2, self.current_number):
            if self.current_number % divisor == 0:
                return True
        return False
