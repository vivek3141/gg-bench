import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum possible value of N
        self.Max_N = 100  # Adjust as needed for game difficulty

        # Define action space: actions are integers from 1 to Max_N - 1
        # Actions correspond to possible proper divisors to subtract
        self.action_space = spaces.Discrete(self.Max_N - 1)  # Actions: 0 to Max_N - 2

        # Define observation space: the current value of N
        self.observation_space = spaces.Box(
            low=1, high=self.Max_N, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Choose a random starting number N between 10 and Max_N inclusive
        self.N = np.random.randint(10, self.Max_N + 1)

        # Player 1 starts the game
        self.current_player = 1

        # Game is not over
        self.done = False

        # Return the initial observation and info dictionary
        return np.array([self.N], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        # Map the action index to the actual proper divisor (action + 1)
        divisor = action + 1

        # Check if the chosen divisor is a valid proper divisor of N
        if divisor < 1 or divisor >= self.N or self.N % divisor != 0:
            # Invalid move: end the game with a penalty
            self.done = True
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        # Valid move: subtract the divisor from N
        self.N -= divisor

        # Check if the current player wins (opponent cannot make a move)
        if self.N == 1 or len(self._get_proper_divisors(self.N)) == 0:
            # Current player wins
            self.done = True
            return np.array([self.N], dtype=np.int32), 1, True, False, {}

        # Switch to the next player
        self.current_player = 3 - self.current_player

        # Continue the game with a negative reward for the step
        return np.array([self.N], dtype=np.int32), -10, False, False, {}

    def render(self):
        # Return a string representation of the current game state
        valid_moves = self.valid_moves()
        return (
            f"Player {self.current_player}'s Turn:\n"
            f"Current N: {self.N}\n"
            f"Valid moves: {valid_moves}\n"
        )

    def valid_moves(self):
        # Return a list of valid moves (proper divisors of N)
        return self._get_proper_divisors(self.N)

    def _get_proper_divisors(self, n):
        # Helper function to calculate proper divisors of n
        return [i for i in range(1, n) if n % i == 0]
