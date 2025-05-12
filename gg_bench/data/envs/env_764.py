import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete actions representing numbers from 2 to 50 (indices 0 to 48)
        self.action_space = spaces.Discrete(49)

        # Observation space:
        # - First 49 elements represent availability of numbers 2 to 50 (1 if available, 0 if taken)
        # - Next 2 elements represent the scores of player 1 and player 2
        self.observation_space = spaces.Box(low=0, high=50, shape=(51,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers from 2 to 50 are available at the start (1 indicates available)
        self.available_numbers = np.ones(49, dtype=np.int32)
        # Scores of Player 1 and Player 2
        self.scores = {1: 0, -1: 0}
        # Player 1 starts the game
        self.current_player = 1
        self.done = False
        # Return the initial observation and an empty info dictionary
        return self._get_observation(), {}

    def step(self, action):
        # Check if the game has already ended
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if the action is within valid range
        if not self.action_space.contains(action):
            return self._get_observation(), -10, True, False, {}

        # Check if the selected number is available
        if self.available_numbers[action] == 0:
            # Invalid move (number not available)
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move
        # Mark the number as taken
        self.available_numbers[action] = 0
        # Map action index to the actual number (2 to 50)
        selected_number = action + 2
        # Calculate the unique prime factors
        prime_factors = self._get_prime_factors(selected_number)
        # Update the current player's score
        points_gained = len(prime_factors)
        self.scores[self.current_player] += points_gained

        # Check for victory condition
        if self.scores[self.current_player] >= 15:
            # Current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Check if no numbers are left
        if np.sum(self.available_numbers) == 0:
            # Game ends in a draw
            self.done = True
            return self._get_observation(), 0, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        # Return the updated observation
        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Visual representation of the game state
        available_numbers = [
            str(i + 2) for i in range(49) if self.available_numbers[i] == 1
        ]
        taken_numbers = [
            str(i + 2) for i in range(49) if self.available_numbers[i] == 0
        ]
        state_str = "Available Numbers: " + ", ".join(available_numbers) + "\n"
        state_str += f"Player 1 Score: {self.scores[1]}\n"
        state_str += f"Player 2 Score: {self.scores[-1]}\n"
        state_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return state_str

    def valid_moves(self):
        # Return a list of valid action indices (available numbers)
        return [i for i in range(49) if self.available_numbers[i] == 1]

    def _get_observation(self):
        # Combine available numbers and scores into a single observation array
        observation = np.concatenate(
            (
                self.available_numbers,
                np.array([self.scores[1], self.scores[-1]], dtype=np.int32),
            )
        )
        return observation

    def _get_prime_factors(self, number):
        # Function to calculate the unique prime factors of a number
        i = 2
        factors = set()
        while i * i <= number:
            if number % i:
                i += 1
            else:
                factors.add(i)
                number //= i
        if number > 1:
            factors.add(number)
        return factors
