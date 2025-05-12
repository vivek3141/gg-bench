import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Numbers from 2 to 50 inclusive
        self.numbers = np.arange(2, 51)  # Array of numbers from 2 to 50

        # Define action space: indices from 0 to 48 corresponding to numbers 2 to 50
        self.action_space = spaces.Discrete(49)

        # Define observation space: array of 49 elements with values -1, 0, 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(49,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.observation = np.zeros(49, dtype=np.int8)  # Observation array

        self.current_player = 1  # Current player (1 or -1)

        self.selected_numbers = {1: [], -1: []}  # Dictionary to track selections

        self.done = False  # Game state

        return self.observation.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.observation.copy(), 0, True, False, {}  # Game already over

        number = action + 2  # Map action index to actual number

        # Get valid actions for current player
        valid_actions = self.valid_moves()

        if action not in valid_actions:
            self.done = True
            return self.observation.copy(), -10, True, False, {}  # Invalid move

        # Valid move: update observation and selections
        self.observation[action] = self.current_player
        self.selected_numbers[self.current_player].append(number)

        # Switch to the next player
        self.current_player *= -1

        # Check if the next player has any valid moves
        if not self.valid_moves():
            self.done = True
            self.current_player *= -1  # Switch back to declaring the winner
            return self.observation.copy(), 1, True, False, {}  # Current player wins

        return self.observation.copy(), 0, False, False, {}  # Continue game

    def render(self):
        output = "Available Numbers:\n"
        for idx, val in enumerate(self.observation):
            if val == 0:
                output += f"{idx + 2} "
        output += "\nYour Selections:\n"
        your_numbers = self.selected_numbers.get(self.current_player * -1, [])
        output += ", ".join(map(str, your_numbers))
        output += "\nOpponent's Selections:\n"
        opponent_numbers = self.selected_numbers.get(self.current_player, [])
        output += ", ".join(map(str, opponent_numbers))
        return output

    def valid_moves(self):
        valid_actions = []

        # Combine selections from both players
        selected_numbers = self.selected_numbers[1] + self.selected_numbers[-1]

        for action in range(49):
            if self.observation[action] != 0:
                continue  # Skip numbers already selected

            number = action + 2

            # Check if the number is a multiple/divisor of any selected number
            is_valid = True
            for selected_number in selected_numbers:
                if number % selected_number == 0 or selected_number % number == 0:
                    is_valid = False
                    break

            if is_valid:
                valid_actions.append(action)

        return valid_actions
