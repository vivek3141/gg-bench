import numpy as np
import gymnasium as gym
from gymnasium import spaces


def get_unique_prime_factors(n):
    factors = set()
    d = 2
    while n > 1 and d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    return factors


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: integers from 0 to 48 corresponding to numbers 2 to 50
        self.action_space = spaces.Discrete(49)

        # Define observation space:
        # - First 49 entries: availability of numbers 2 to 50 (1.0 if available, 0.0 if not)
        # - Next 2 entries: scores of Player 1 and Player 2
        # - Last entry: current player (1.0 for Player 1, -1.0 for Player 2)
        low_obs = np.array([0.0] * 49 + [0.0, 0.0] + [-1.0], dtype=np.float32)
        high_obs = np.array([1.0] * 49 + [np.inf, np.inf] + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )

        # Precompute prime factors for numbers from 2 to 50
        self.prime_factors = {n: get_unique_prime_factors(n) for n in range(2, 51)}

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize number pool: True if the number is available, False if taken
        self.number_pool = {n: True for n in range(2, 51)}
        # Initialize scores for Player 1 (1) and Player 2 (-1)
        self.scores = {1: 0, -1: 0}
        # Set current player: 1 for Player 1, -1 for Player 2
        self.current_player = 1
        # Game is not over
        self.done = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        number = action + 2  # Map action index to number

        # Check if the number is available
        if not self.number_pool.get(number, False):
            # Invalid move
            return self._get_observation(), -10.0, False, False, {}

        # Valid move
        self.number_pool[number] = False  # Remove number from pool

        # Calculate points gained
        prime_factors = self.prime_factors[number]
        points_gained = len(prime_factors)
        self.scores[self.current_player] += points_gained

        # Check for winning condition
        if (
            self.scores[self.current_player] >= 10
            and self.scores[self.current_player] > self.scores[-self.current_player]
        ):
            # Current player wins
            self.done = True
            return self._get_observation(), 1.0, True, False, {}

        # Check if all numbers are exhausted
        if not any(self.number_pool.values()):
            self.done = True
            # Determine winner based on scores
            if self.scores[1] > self.scores[-1]:
                winner = 1
            elif self.scores[-1] > self.scores[1]:
                winner = -1
            else:
                winner = 0
            if winner == self.current_player:
                return self._get_observation(), 1.0, True, False, {}
            elif winner == 0:
                return self._get_observation(), 0.0, True, False, {}
            else:
                return self._get_observation(), -1.0, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        return self._get_observation(), 0.0, False, False, {}

    def render(self):
        output = "Current Game State:\n"
        output += (
            f"Numbers available: {[n for n in range(2, 51) if self.number_pool[n]]}\n"
        )
        output += f"Player 1 Score: {self.scores[1]}\n"
        output += f"Player 2 Score: {self.scores[-1]}\n"
        output += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return output

    def valid_moves(self):
        return [n - 2 for n in range(2, 51) if self.number_pool[n]]

    def _get_observation(self):
        # First 49 entries: availability of numbers 2 to 50
        numbers_availability = np.array(
            [1.0 if self.number_pool[n] else 0.0 for n in range(2, 51)],
            dtype=np.float32,
        )
        # Next two entries: scores of Player 1 and Player 2
        scores = np.array([self.scores[1], self.scores[-1]], dtype=np.float32)
        # Last entry: current player (1.0 or -1.0)
        current_player = np.array([float(self.current_player)], dtype=np.float32)
        # Combine into one observation array
        observation = np.concatenate((numbers_availability, scores, current_player))
        return observation
