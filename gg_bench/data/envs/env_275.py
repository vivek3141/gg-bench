import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 15 possible actions (positions 0-2, digits 1-5)
        # Action mapping: action = position * 5 + (new_digit - 1)
        self.action_space = spaces.Discrete(15)

        # Define observation space
        # Observation consists of the safe combination (3 digits) and the player's secret code (3 digits)
        # All digits range from 1 to 5 inclusive
        self.observation_space = spaces.Box(low=1, high=5, shape=(6,), dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the safe combination
        self.safe_combination = np.array([1, 1, 1], dtype=np.int32)

        # Randomly generate secret codes for both players (digits 1-5)
        self.player1_code = self.np_random.integers(1, 6, size=(3,))
        self.player2_code = self.np_random.integers(1, 6, size=(3,))

        # Player 1 starts
        self.current_player = 1
        self.turn_number = 1  # Keep track of turns to enforce first-turn rule
        self.done = False

        # Return the initial observation and info
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            # Invalid move after game has ended
            return self._get_observation(), -10, True, False, {}

        # Map action to position and new_digit
        position = action // 5  # Position in [0, 1, 2]
        new_digit = (action % 5) + 1  # Digit in [1, 5]

        # Apply the action: change one digit of the safe combination
        self.safe_combination[position] = new_digit

        # Get the current player's secret code
        player_code = (
            self.player1_code if self.current_player == 1 else self.player2_code
        )

        # Check for win condition (cannot win on first turn)
        if self.turn_number > 1 and np.array_equal(self.safe_combination, player_code):
            # Current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # No win; switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1
        self.turn_number += 1

        # Continue the game
        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Provide a string representation of the current state
        output = f"Current Safe Combination: {self.safe_combination.tolist()}\n"
        output += f"Player {self.current_player}'s Turn\n"
        output += (
            f"Player {self.current_player}'s Secret Code: {self._get_player_code()}\n"
        )
        return output

    def valid_moves(self):
        # Return all possible actions if the game is not over
        return list(range(15)) if not self.done else []

    def _get_observation(self):
        # Construct the observation for the current player
        player_code = (
            self.player1_code if self.current_player == 1 else self.player2_code
        )
        observation = np.concatenate((self.safe_combination, player_code))
        return observation

    def _get_player_code(self):
        # Helper method to get the current player's secret code as a list
        return (
            self.player1_code if self.current_player == 1 else self.player2_code
        ).tolist()
