import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Constants
        self.MAX_NUMBER = 100  # Maximum number allowed in the game

        # Define action and observation space
        self.action_space = spaces.Discrete(
            self.MAX_NUMBER + 1
        )  # Actions range from 0 to MAX_NUMBER

        # Observation space: [current_number_normalized, current_player] + used_numbers[1..MAX_NUMBER]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.MAX_NUMBER + 2,),
            dtype=np.float32,
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the starting number
        self.starting_number = np.random.randint(10, self.MAX_NUMBER + 1)
        self.current_number = self.starting_number

        # Initialize used numbers list
        self.used_numbers = np.zeros(self.MAX_NUMBER + 1, dtype=np.int8)

        # Mark the starting number as used
        self.used_numbers[self.starting_number] = 1

        # Set current player (1 or -1)
        self.current_player = 1

        # Game over flag
        self.done = False

        # Prepare the observation
        observation = self._get_observation()

        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if the action is within valid range
        if action < 2 or action >= self.current_number:
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        # Check if action is a proper divisor of current_number
        if self.current_number % action != 0:
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        # Check if the number has been used before
        if self.used_numbers[action] == 1:
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        # Valid action; update the game state
        self.used_numbers[action] = 1  # Mark the action as used
        self.current_number = action  # Update current number

        # Check for win condition
        valid_divisors = self._get_valid_divisors()
        if not valid_divisors:
            # Opponent has no valid moves; current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, self.done, False, {}

        # Continue the game
        reward = 0
        self.current_player *= -1  # Switch player
        return self._get_observation(), reward, self.done, False, {}

    def render(self):
        used_numbers_list = [
            str(i) for i in range(2, self.MAX_NUMBER + 1) if self.used_numbers[i]
        ]
        used_numbers_str = ", ".join(used_numbers_list)
        return (
            f"Current Number: {self.current_number}\n"
            f"Used Numbers: {used_numbers_str}\n"
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        )

    def valid_moves(self):
        return self._get_valid_divisors()

    def _get_observation(self):
        # Normalize current number
        current_number_normalized = self.current_number / self.MAX_NUMBER

        # Current player (1 or -1)
        current_player = self.current_player

        # Used numbers status (from index 2 to MAX_NUMBER)
        used_numbers_status = self.used_numbers[2:].astype(np.float32)

        # Combine into a single observation array
        observation = np.concatenate(
            (
                np.array([current_number_normalized], dtype=np.float32),
                np.array([current_player], dtype=np.float32),
                used_numbers_status,
            )
        )
        return observation

    def _get_valid_divisors(self):
        # Find proper divisors greater than 1 and less than current number
        potential_divisors = [
            i
            for i in range(2, self.current_number)
            if self.current_number % i == 0 and self.used_numbers[i] == 0
        ]
        return potential_divisors
