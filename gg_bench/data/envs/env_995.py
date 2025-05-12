import numpy as np
import gymnasium as gym
from gymnasium import spaces


def is_prime(n):
    """
    Check if a number is prime.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    w = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += w
        w = 6 - w
    return True


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: numbers 1 to 10 inclusive, represented as actions 0 to 9
        self.action_space = spaces.Discrete(10)

        # Observation space: the total sum, ranging from 0 to 1000
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(1,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.total_sum = 0
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False

        return np.array([self.total_sum], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # Game is over
            return np.array([self.total_sum], dtype=np.int32), 0, True, False, {}

        if action < 0 or action > 9:
            # Invalid action
            reward = -10
            self.done = True
            return np.array([self.total_sum], dtype=np.int32), reward, True, False, {}

        # Map action 0-9 to number 1-10
        number_added = action + 1

        # Add the chosen number to the total sum
        self.total_sum += number_added

        if is_prime(self.total_sum):
            # Current player loses
            reward = -10
            self.done = True
            return np.array([self.total_sum], dtype=np.int32), reward, True, False, {}
        else:
            # Continue the game
            reward = 0
            self.current_player *= -1  # Switch player
            return np.array([self.total_sum], dtype=np.int32), reward, False, False, {}

    def render(self):
        """
        Return a string representation of the game state.
        """
        return f"Total Sum: {self.total_sum}, Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        """
        Return a list of valid moves (indices of the action_space).
        """
        if self.done:
            return []
        else:
            return list(range(10))  # Actions 0 to 9 correspond to numbers 1 to 10
