import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Primes from 2 to 29
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        # Action space: picking one of the 10 primes
        self.action_space = spaces.Discrete(10)

        # Observation space: [Player1_score, Player2_score, primes_availability]
        # Shape: (12,)
        low = np.array([0, 0] + [0] * 10, dtype=np.float32)
        high = np.array([50, 50] + [1] * 10, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.scores = [0, 0]
        self.primes_available = [1] * 10  # All primes are available

        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False

        # Create observation
        observation = np.array(self.scores + self.primes_available, dtype=np.float32)

        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        if action < 0 or action >= len(self.primes):
            # Invalid action index
            self.done = True
            return self._get_obs(), -10, True, False, {}

        if self.primes_available[action] == 0:
            # Prime not available, invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid action, proceed
        selected_prime = self.primes[action]
        self.primes_available[action] = 0  # Remove prime from pool

        # Update score
        self.scores[self.current_player] += selected_prime

        reward = -10  # Default reward for a valid move

        # Check for win or loss
        if self.scores[self.current_player] == 50:
            # Current player wins
            self.done = True
            reward = 1
        elif self.scores[self.current_player] > 50:
            # Current player loses
            self.done = True
            reward = -1
        else:
            # Game continues
            # Switch player
            self.current_player = 1 - self.current_player

        observation = self._get_obs()

        return observation, reward, self.done, False, {}

    def render(self):
        # Return a string representation of the game state
        player = self.current_player + 1
        primes_remaining = [
            p for p, available in zip(self.primes, self.primes_available) if available
        ]
        state_str = f"Player {player}'s turn.\n"
        state_str += f"Scores: Player 1: {self.scores[0]}, Player 2: {self.scores[1]}\n"
        state_str += f"Available primes: {', '.join(map(str, primes_remaining))}\n"
        return state_str

    def valid_moves(self):
        # Return indices of available primes
        return [i for i, available in enumerate(self.primes_available) if available]

    def _get_obs(self):
        observation = np.array(self.scores + self.primes_available, dtype=np.float32)
        return observation
