import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Numbers from 1 to 9
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            9, dtype=np.int8
        )  # Numbers 1 to 9 are available
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}  # Game is over

        if action < 0 or action >= 9 or self.available_numbers[action] == 0:
            # Invalid action
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Invalid move ends the game

        chosen_number = action + 1  # Map action (0-8) to number (1-9)

        # Remove selected number and its associates
        numbers_to_remove = self._get_numbers_to_remove(chosen_number)

        # Update available numbers
        for num in numbers_to_remove:
            index = num - 1  # Map number to index
            self.available_numbers[index] = 0

        # Check if the game is over
        if np.sum(self.available_numbers) == 0:
            # Current player picked the last number and loses
            self.done = True
            reward = -10  # Current player loses
            return self._get_observation(), reward, True, False, {}

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Game continues
        return self._get_observation(), 0, False, False, {}

    def render(self):
        available_numbers = [
            str(i + 1) if self.available_numbers[i] == 1 else "X" for i in range(9)
        ]
        board_str = "Available Numbers: [" + ", ".join(available_numbers) + "]"
        return board_str

    def valid_moves(self):
        return [i for i in range(9) if self.available_numbers[i] == 1]

    def _get_observation(self):
        return self.available_numbers.copy()

    def _get_numbers_to_remove(self, chosen_number):
        numbers_to_remove = set()
        numbers_to_remove.add(chosen_number)  # Remove the chosen number

        # Remove factors (excluding 1 unless 1 is chosen)
        if chosen_number == 1:
            pass  # Remove only number 1
        else:
            for i in range(2, chosen_number):
                if chosen_number % i == 0 and self.available_numbers[i - 1]:
                    numbers_to_remove.add(i)

        # Remove multiples of the chosen number
        for multiple in range(chosen_number * 2, 10, chosen_number):
            if self.available_numbers[multiple - 1]:
                numbers_to_remove.add(multiple)

        return numbers_to_remove
