import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(19): numbers 2 to 20
        self.action_space = spaces.Discrete(19)

        # The observation space is a Box with shape (20,):
        # - First 19 entries represent the availability of numbers 2 to 20 (1 for available, 0 for not available)
        # - The last entry is the last number selected by the opponent (0 if no number has been selected yet)
        self.observation_space = spaces.Box(
            low=0, high=20, shape=(20,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the list of available numbers (1 for available, 0 for not available)
        self.available_numbers = np.ones(19, dtype=np.float32)

        # The last number selected by the opponent (0 if no number has been selected yet)
        self.last_opponent_number = 0

        # Current player: 1 or -1 (not used in this implementation, but can be helpful)
        self.current_player = 1

        # Game over flag
        self.done = False

        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        # If the game is already over
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Map action to the selected number (0 -> 2, 1 -> 3, ..., 18 -> 20)
        selected_number = action + 2

        # Check if the selected number is available
        if self.available_numbers[action] == 0:
            # Invalid move: number already taken
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Check if the move is valid according to the game rules
        if self.last_opponent_number == 0:
            # First move: any number is valid
            valid_move = True
        else:
            # The selected number must share a common factor >1 with the last opponent number
            if math.gcd(selected_number, self.last_opponent_number) > 1:
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move according to the game rules
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Valid move: update the game state
        self.available_numbers[action] = 0  # Mark the number as not available

        # Update the last opponent number (for the next player's turn)
        self.last_opponent_number = selected_number

        # Switch to the other player (not necessary for self-play, but kept for clarity)
        self.current_player *= -1

        # Check if the next player has any valid moves
        if not self.valid_moves():
            # Next player cannot make a valid move: current player wins
            self.done = True
            reward = 1  # Current player wins
            return self._get_obs(), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        # Construct the observation array
        # - First 19 entries: availability of numbers 2 to 20
        # - Last entry: last opponent number
        observation = np.concatenate(
            (self.available_numbers, [self.last_opponent_number])
        )
        return observation

    def render(self):
        # Generate a string representation of the game state
        number_list = [
            str(i + 2) if self.available_numbers[i] == 1 else "X" for i in range(19)
        ]
        board_str = "Available Numbers: " + ", ".join(number_list)
        board_str += "\nLast Opponent Number: " + str(self.last_opponent_number)
        return board_str

    def valid_moves(self):
        # Return a list of valid moves (action indices) for the current player
        valid_moves = []
        for i in range(19):
            if self.available_numbers[i] == 1:
                number = i + 2
                # First move or number shares a common factor >1 with the last opponent number
                if (
                    self.last_opponent_number == 0
                    or math.gcd(number, self.last_opponent_number) > 1
                ):
                    valid_moves.append(i)
        return valid_moves
