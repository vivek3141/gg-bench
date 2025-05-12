import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 90 possible two-digit numbers from 10 to 99 inclusive
        self.action_space = spaces.Discrete(90)

        # Observation space:
        # - First 90 elements: 1 if the number (from 10 to 99) has been used, 0 if not
        # - Last element: the required starting digit for the next number (0-9)
        self.observation_space = spaces.Box(low=0, high=9, shape=(91,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.used_numbers = np.zeros(90, dtype=np.int8)  # Numbers from 10 to 99
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Randomly select a starting number between 10 and 99
        start_index = self.np_random.integers(0, 90)
        self.start_number = start_index + 10
        self.used_numbers[start_index] = 1

        self.required_start_digit = (
            self.start_number % 10
        )  # Last digit of the starting number

        # Prepare the observation
        observation = np.append(self.used_numbers, self.required_start_digit)
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                np.append(self.used_numbers, self.required_start_digit),
                0,
                True,
                False,
                {},
            )  # Game over

        # Map action index to the actual number (from 10 to 99)
        number = action + 10

        # Check if the action is valid
        valid = True
        if number < 10 or number > 99:
            valid = False
        elif self.used_numbers[action] == 1:
            valid = False
        elif number // 10 != self.required_start_digit:
            valid = False

        if not valid:
            # Invalid move
            self.done = True
            return (
                np.append(self.used_numbers, self.required_start_digit),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid move
        self.used_numbers[action] = 1  # Mark the number as used
        self.required_start_digit = number % 10  # Update required starting digit

        # Check if the opponent can make a move
        opponent_valid_moves = self._get_valid_moves()

        if not opponent_valid_moves:
            # Current player wins
            self.done = True
            return (
                np.append(self.used_numbers, self.required_start_digit),
                1,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player *= -1

        # Reward for a valid move
        return (
            np.append(self.used_numbers, self.required_start_digit),
            -10,
            False,
            False,
            {},
        )

    def render(self):
        used_numbers = [i + 10 for i, used in enumerate(self.used_numbers) if used == 1]
        state_str = f"Used Numbers: {used_numbers}\n"
        state_str += f"Required Starting Digit: {self.required_start_digit}\n"
        state_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return state_str

    def valid_moves(self):
        # Return a list of action indices of valid moves
        valid_moves = []
        for action in range(90):
            number = action + 10
            if (
                self.used_numbers[action] == 0
                and number // 10 == self.required_start_digit
            ):
                valid_moves.append(action)
        return valid_moves

    def _get_valid_moves(self):
        # Helper function to get valid moves for the current required_start_digit
        valid_moves = []
        for action in range(90):
            number = action + 10
            if (
                self.used_numbers[action] == 0
                and number // 10 == self.required_start_digit
            ):
                valid_moves.append(action)
        return valid_moves
