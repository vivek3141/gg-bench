import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers from 1 to 100 (represented as 0 to 99 in action space)
        self.action_space = spaces.Discrete(100)

        # Define observation space:
        # - observations[0]: last number played (1 to 100)
        # - observations[1]: current player (1 or -1)
        # - observations[2:]: numbers used (0: not used, 1: used by Player 1, -1: used by Player 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(102,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_number = 1
        self.current_player = 1  # Player 1 starts (1), Player 2 is represented as -1
        self.numbers_used = np.zeros(
            100, dtype=np.int8
        )  # Index 0 corresponds to number 1
        self.done = False
        return self._get_obs(), {}  # Return initial observation and empty info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game already over

        # Convert action to actual number (1 to 100)
        number = action + 1

        # Check if the move is valid
        if not self._is_valid_move(number):
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid move, player loses

        # Update the game state with the valid move
        self.last_number = number
        self.numbers_used[number - 1] = (
            self.current_player
        )  # Mark number as used by current player

        # Check if the next player has any valid moves
        self.current_player *= -1  # Switch player
        if not self._has_valid_move():
            self.done = True
            return self._get_obs(), 1, True, False, {}  # Current player wins

        # If the game continues
        return self._get_obs(), 0, False, False, {}  # No reward, game continues

    def render(self):
        # Generate a string representation of the current game state
        player = "Player 1" if self.current_player == 1 else "Player 2"
        used_numbers = (
            np.where(self.numbers_used != 0)[0] + 1
        )  # Get the list of used numbers
        render_str = "--- Factor Chain Game ---\n"
        render_str += f"Current Sequence ends with: {self.last_number}\n"
        render_str += f"Current Player: {player}\n"
        render_str += f"Used Numbers: {used_numbers.tolist()}\n"
        return render_str

    def valid_moves(self):
        # Return a list of valid moves (indices in action_space) for the current player
        valid_moves = []
        for action in range(100):
            number = action + 1
            if self._is_valid_move(number):
                valid_moves.append(action)
        return valid_moves

    def _get_obs(self):
        # Construct the observation array
        obs = np.zeros(102, dtype=np.int8)
        obs[0] = self.last_number
        obs[1] = self.current_player
        obs[2:] = self.numbers_used
        return obs

    def _is_valid_move(self, number):
        # Check if the number is within the valid range
        if number < 1 or number > 100:
            return False

        # Check if the number is the same as the last number played
        if number == self.last_number:
            return False

        # Check if the number has already been used by the current player
        if self.numbers_used[number - 1] == self.current_player:
            return False

        # Check if the number is a factor or multiple of the last number
        if self.last_number % number == 0 or number % self.last_number == 0:
            return True

        return False

    def _has_valid_move(self):
        # Check if the current player has any valid moves available
        for number in range(1, 101):
            if self._is_valid_move(number):
                return True
        return False
