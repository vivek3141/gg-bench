import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to adding integers from 1 to 10 (indices 0 to 9)
        self.action_space = spaces.Discrete(10)

        # Observation is the current shared number
        self.observation_space = spaces.Box(
            low=0, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = 2  # Starting number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.shared_number], dtype=np.int32), {}

    def step(self, action):
        addition = action + 1  # Map action index to addition (1 to 10)

        if addition < 1 or addition > 10:
            # Invalid action (should not happen due to action_space definition)
            return (
                np.array([self.shared_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        new_number = self.shared_number + addition

        if not self.is_prime(new_number):
            # Move results in non-prime number, player loses
            return (
                np.array([self.shared_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Update the shared number
        self.shared_number = new_number

        # Check if the next player has any valid moves
        if not self.has_valid_moves():
            # Opponent has no valid moves, current player wins
            return (
                np.array([self.shared_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player *= -1

        return (
            np.array([self.shared_number], dtype=np.int32),
            0,
            False,
            False,
            {},
        )

    def render(self):
        return f"Current Shared Number: {self.shared_number}"

    def valid_moves(self):
        # Return list of valid action indices (0 to 9)
        valid_actions = []
        for action in range(10):
            addition = action + 1
            potential_number = self.shared_number + addition
            if self.is_prime(potential_number):
                valid_actions.append(action)
        return valid_actions

    def has_valid_moves(self):
        # Check if there are any valid moves for the next player
        for action in range(10):
            addition = action + 1
            potential_number = self.shared_number + addition
            if self.is_prime(potential_number):
                return True
        return False

    @staticmethod
    def is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
