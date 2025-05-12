import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 possible numbers to choose from: numbers 2 to 10 inclusive
        self.action_space = spaces.Discrete(9)

        # Observation space consists of:
        # - First 9 elements: availability of numbers 2 to 10 (0: available, 1: taken)
        # - 10th element: opponent's last chosen number (2 to 10, or -1 if none)
        self.observation_space = spaces.Box(low=-1, high=10, shape=(10,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the number pool (numbers 2 to 10)
        self.number_pool = np.array(range(2, 11), dtype=np.int8)

        # Tracking availability of numbers (0: available, 1: taken)
        self.number_taken = np.zeros(9, dtype=np.int8)

        # Opponent's last number (-1 indicates no previous number)
        self.opponent_last_number = -1

        # Current player's last number (-1 indicates no previous number)
        self.current_last_number = -1

        # Start with Player 1 (represented as 1), Player 2 is -1
        self.current_player = 1

        # Game over flag
        self.done = False

        # No need for truncated in the new gymnasium API
        # Return the initial observation and an empty info dict
        return self._get_observation(), {}

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Convert action index to actual number (numbers 2 to 10)
        selected_number = action + 2

        # Check if the selected number is available
        if self.number_taken[action] == 1:
            # Invalid move: number already taken
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if it's the first turn (no opponent's last number)
        if self.opponent_last_number == -1:
            valid = True  # Any number can be chosen
        else:
            # Check if the selected number is neither a factor nor a multiple of opponent's last number
            if self.is_factor_or_multiple(selected_number, self.opponent_last_number):
                # Invalid move
                self.done = True
                return self._get_observation(), -10, True, False, {}

            valid = True

        if valid:
            # Valid move, update the game state
            self.number_taken[action] = 1  # Mark the number as taken
            self.number_pool = self.number_pool[self.number_pool != selected_number]
            self.current_last_number = selected_number

            # Check if the opponent has any valid moves
            self.current_player *= -1  # Switch to opponent
            opponent_valid_moves = self.valid_moves()
            self.current_player *= -1  # Switch back to current player

            if not opponent_valid_moves:
                # Opponent has no valid moves, current player wins
                self.done = True
                return self._get_observation(), 1, True, False, {}
            else:
                # Game continues, switch players
                self.opponent_last_number = self.current_last_number
                self.current_last_number = -1
                self.current_player *= -1  # Switch to opponent
                return self._get_observation(), 0, False, False, {}

        else:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

    def render(self):
        # Provide a simple text-based representation of the game state
        output = "=== Factor Frenzy ===\n\n"
        output += f"Number Pool: {[num for i, num in enumerate(range(2, 11)) if self.number_taken[i] == 0]}\n\n"
        output += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        output += f"Opponent's Last Number: {self.opponent_last_number if self.opponent_last_number != -1 else 'None'}\n"
        output += f"Your Last Number: {self.current_last_number if self.current_last_number != -1 else 'None'}\n"
        output += f"Numbers Taken: {[(num, 'Player 1' if self.current_player == -1 else 'Player 2') for i, num in enumerate(range(2, 11)) if self.number_taken[i] == 1]}\n"
        return output

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        valid_actions = []
        for action in range(9):
            if self.number_taken[action] == 0:
                selected_number = action + 2
                if self.opponent_last_number == -1:
                    # First turn, any number is valid
                    valid_actions.append(action)
                elif not self.is_factor_or_multiple(
                    selected_number, self.opponent_last_number
                ):
                    # Number is valid if it's not a factor or multiple of opponent's last number
                    valid_actions.append(action)
        return valid_actions

    def is_factor_or_multiple(self, num1, num2):
        # Check if num1 is a factor or multiple of num2
        if num1 == 0 or num2 == 0:
            return False  # Avoid division by zero
        if num2 % num1 == 0 or num1 % num2 == 0:
            return True
        return False

    def _get_observation(self):
        # Return the observation
        obs = np.zeros(10, dtype=np.int8)
        obs[:9] = self.number_taken  # First 9 elements: availability of numbers 2 to 10
        obs[9] = (
            self.opponent_last_number
        )  # 10th element: opponent's last number (-1 if none)
        return obs
