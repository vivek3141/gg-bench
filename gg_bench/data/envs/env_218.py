import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Game parameters
        self.target_value = 20  # Can be modified for different difficulty levels
        self.max_total = (
            2 * self.target_value + 10
        )  # Arbitrary max total to define observation space
        self.available_numbers = list(range(1, 11))  # Numbers from 1 to 10

        # Define action and observation spaces
        # Actions 0-9 correspond to selecting numbers 1-10 respectively
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(
            low=0, high=self.max_total, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_total = 0
        self.current_player = 1  # Can be 1 or -1 to represent the two players
        self.done = False

        return np.array([self.shared_total], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return (
                np.array([self.shared_total], dtype=np.int32),
                -10,  # Penalty for action after game is over
                True,
                False,
                {},
            )

        if not self.action_space.contains(action):
            # Invalid action (not in action_space)
            self.done = True
            return (
                np.array([self.shared_total], dtype=np.int32),
                -10,  # Penalty for invalid action
                True,
                False,
                {},
            )

        # Map action to the selected number (1-10)
        selected_number = action + 1

        # Update shared total
        self.shared_total += selected_number

        # Check winning condition
        if self.is_prime(self.shared_total) and self.shared_total > self.target_value:
            # Current player wins
            self.done = True
            reward = 1  # Reward for winning
            return (
                np.array([self.shared_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check losing condition
        if self.shared_total > 2 * self.target_value:
            # Current player loses
            self.done = True
            reward = -1  # Penalty for losing
            return (
                np.array([self.shared_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Game continues; switch to the other player
        self.current_player *= -1

        return (
            np.array([self.shared_total], dtype=np.int32),  # Updated observation
            0,  # No reward
            False,  # Game is not over
            False,
            {},
        )

    def render(self):
        return f"Shared Total: {self.shared_total}"

    def valid_moves(self):
        # All numbers from 1 to 10 are valid moves
        # Actions are indices 0-9 corresponding to numbers 1-10
        return list(range(10))

    def is_prime(self, n):
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
