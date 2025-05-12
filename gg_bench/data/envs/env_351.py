import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The numbers 1 to 9 correspond to actions 0 to 8
        self.action_space = spaces.Discrete(9)
        # Observation is a vector of size 9, 1 if number is available, 0 if not
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers from 1 to 9 are available at the start
        self.available_numbers = [i for i in range(1, 10)]
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.truncated = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), -10, self.done, self.truncated, {}

        # Map action to number (actions 0-8 correspond to numbers 1-9)
        selected_number = action + 1

        # Check if the selected number is valid (available)
        if selected_number not in self.available_numbers:
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid move

        # Remove the selected number and its factors and multiples
        numbers_to_remove = self._get_factors_multiples(selected_number)
        self.available_numbers = [
            num for num in self.available_numbers if num not in numbers_to_remove
        ]

        # Check for game over conditions
        if len(self.available_numbers) == 0:
            # Current player picked the last number and loses
            self.done = True
            return self._get_obs(), -10, True, False, {}
        elif len(self.available_numbers) == 1:
            # Next player is forced to pick the last number and will lose
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch turns to the next player
        self.current_player = 1 if self.current_player == 2 else 2
        return self._get_obs(), -10, False, False, {}  # Valid move

    def render(self):
        # Return a string representation of the current state
        remaining_numbers = " ".join(str(num) for num in self.available_numbers)
        render_str = f"Numbers remaining: {remaining_numbers}\n"
        render_str += f"Player {self.current_player}'s turn.\n"
        return render_str

    def valid_moves(self):
        # Return a list of valid moves (indices of available numbers)
        return [num - 1 for num in self.available_numbers]

    def _get_obs(self):
        # Observation is a vector indicating which numbers are available
        obs = np.zeros(9, dtype=np.int8)
        for num in self.available_numbers:
            obs[num - 1] = 1
        return obs

    def _get_factors_multiples(self, number):
        # Get all factors and multiples of the selected number in the available numbers
        factors_multiples = [number]
        for num in self.available_numbers:
            if num != number:
                if number % num == 0 or num % number == 0:
                    factors_multiples.append(num)
        return factors_multiples
