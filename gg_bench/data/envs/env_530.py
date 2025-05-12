import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete(29), corresponds to numbers 2 to 30
        self.action_space = spaces.Discrete(29)

        # Observation space consists of:
        # - available_numbers: 29 elements (1 if available, 0 if not)
        # - scores: 2 elements (Player 1 score, Player 2 score)
        # - current_player: 1 element (1 or 2)
        # Total of 29 + 2 + 1 = 32 elements
        self.observation_space = spaces.Box(low=0, high=15, shape=(32,), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(29, dtype=np.int32)  # Numbers from 2 to 30
        self.scores = np.zeros(2, dtype=np.int32)  # Player 1 and Player 2 scores
        self.current_player = 1  # Player 1 starts
        self.terminated = False
        observation = self._get_observation()
        return observation, {}  # observation, info

    def _get_observation(self):
        obs = np.concatenate(
            [
                self.available_numbers,
                self.scores,
                np.array([self.current_player], dtype=np.int32),
            ]
        )
        return obs

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, {}

        if not 0 <= action < 29:
            # Invalid action number
            self.terminated = True
            return self._get_observation(), -10, True, False, {}

        if self.available_numbers[action] == 0:
            # Number already selected, invalid move
            self.terminated = True
            return self._get_observation(), -10, True, False, {}

        # Perform the action
        number = action + 2  # Map action to number from 2 to 30
        points = self._calculate_points(number)

        # Update scores
        self.scores[self.current_player - 1] += points

        # Remove the selected number from available numbers
        self.available_numbers[action] = 0

        # Check for win condition
        if self.scores[self.current_player - 1] >= 10:
            self.terminated = True
            return self._get_observation(), 1, True, False, {}

        # Check if all numbers are exhausted
        if np.sum(self.available_numbers) == 0:
            self.terminated = True
            # Determine winner
            if self.scores[0] > self.scores[1]:
                winner = 1
            elif self.scores[1] > self.scores[0]:
                winner = 2
            else:
                # Game ends in a tie
                winner = 0  # Tie
            if winner == self.current_player:
                return self._get_observation(), 1, True, False, {}
            elif winner == 0:
                return self._get_observation(), 0, True, False, {}
            else:
                return self._get_observation(), -1, True, False, {}

        # Switch to next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Return next observation
        return self._get_observation(), 0, False, False, {}

    def _calculate_points(self, number):
        prime_factors = self._prime_factors(number)
        unique_primes = set(prime_factors)
        return len(unique_primes)

    def _prime_factors(self, n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    def render(self):
        available_numbers = [i + 2 for i in range(29) if self.available_numbers[i] == 1]
        s = f"Current Player: {self.current_player}\n"
        s += f"Available Numbers: {available_numbers}\n"
        s += f"Scores: Player 1 = {self.scores[0]}, Player 2 = {self.scores[1]}\n"
        return s

    def valid_moves(self):
        return [i for i in range(29) if self.available_numbers[i] == 1]
