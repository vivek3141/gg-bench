import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 29 possible numbers from 2 to 30 inclusive
        self.action_space = spaces.Discrete(29)
        self.observation_space = spaces.Box(low=0, high=1, shape=(29,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the numbers from 2 to 30 (inclusive)
        self.numbers = np.ones(
            29, dtype=np.int8
        )  # 1 represents available, 0 represents eliminated
        self.current_player = 1  # Player 1 starts
        self.done = False

        return self.numbers.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is already over
            return self.numbers.copy(), -10, True, False, {}

        if not self.is_valid_action(action):
            # Invalid action
            self.done = True
            return self.numbers.copy(), -10, True, False, {}

        # Valid action
        selected_number = action + 2  # Map action to number between 2 and 30
        self.eliminate_numbers(selected_number)

        # Check if current player picked the last number
        if np.sum(self.numbers) == 0:
            # Current player wins
            self.done = True
            reward = 1  # Winning reward
            return self.numbers.copy(), reward, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        # Check if the opponent has any valid moves
        if len(self.valid_moves()) == 0:
            # Opponent cannot move; current player wins
            self.done = True
            reward = 1  # Winning reward
            return self.numbers.copy(), reward, True, False, {}

        # Game continues
        reward = -10  # Penalty for a valid move
        return (
            self.numbers.copy(),
            reward,
            False,
            False,
            {},
        )  # observation, reward, terminated, truncated, info

    def render(self):
        # Return a visual representation of the current state
        available_numbers = [
            str(i + 2) for i, val in enumerate(self.numbers) if val == 1
        ]
        return "Available Numbers: " + ", ".join(available_numbers)

    def valid_moves(self):
        # Return a list of valid action indices
        return [i for i, val in enumerate(self.numbers) if val == 1]

    def is_valid_action(self, action):
        # Check if the action is within the valid range and the number is available
        return 0 <= action < 29 and self.numbers[action] == 1

    def eliminate_numbers(self, selected_number):
        # Eliminate the selected number and its multiples
        for i in range(29):
            number = i + 2
            if self.numbers[i] == 1 and number % selected_number == 0:
                self.numbers[i] = 0
