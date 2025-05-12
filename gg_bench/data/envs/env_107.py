import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: remove a number 1-9 (indices 0-8)
        self.action_space = spaces.Discrete(9)
        # Observation space:
        # 'numbers': status of numbers 1-9 (1: available, 0: removed)
        # 'last_removed': last number removed (0 if no number removed yet)
        self.observation_space = spaces.Dict(
            {
                "numbers": spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8),
                "last_removed": spaces.Discrete(
                    10
                ),  # numbers 0 (no previous move) to 9
            }
        )

        # Initialize game state variables
        self.numbers = None
        self.last_removed = None
        self.current_player = None
        self.done = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.numbers = np.ones(
            9, dtype=np.int8
        )  # All numbers are available at the start
        self.last_removed = 0  # No number has been removed yet
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = {
            "numbers": self.numbers.copy(),
            "last_removed": self.last_removed,
        }
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_obs(), 0, True, False, {}

        number = action + 1  # Convert action index to number (1-9)

        # Check if the number is available
        if self.numbers[action] == 0:
            # Invalid move: number already removed
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Check if the move is valid according to the game's rules
        if self.last_removed != 0:
            gcd = math.gcd(number, self.last_removed)
            if gcd > 1:
                # Invalid move: shares common factor >1 with last removed number
                self.done = True
                return self._get_obs(), -10, True, False, {}

        # Valid move
        self.numbers[action] = 0  # Remove the number
        self.last_removed = number  # Update last removed number

        # Switch current player
        self.current_player = -self.current_player

        # Check if the next player has any valid moves
        opponent_valid_moves = self._get_valid_moves()

        if not opponent_valid_moves:
            # Opponent has no valid moves, current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Game continues
        return self._get_obs(), 0, False, False, {}

    def render(self):
        output = "Current Number List: ["
        output += ",".join(
            str(i + 1) if self.numbers[i] == 1 else "X" for i in range(9)
        )
        output += "]\n"
        output += f"Last number removed: {self.last_removed}\n"
        return output

    def valid_moves(self):
        """Return a list of valid action indices for the current player."""
        return self._get_valid_moves()

    def _get_obs(self):
        observation = {
            "numbers": self.numbers.copy(),
            "last_removed": self.last_removed,
        }
        return observation

    def _get_valid_moves(self):
        valid_moves = []
        for i in range(9):
            if self.numbers[i] == 1:
                number = i + 1
                if self.last_removed == 0:
                    # No previous move, all available numbers are valid
                    valid_moves.append(i)
                else:
                    gcd = math.gcd(number, self.last_removed)
                    if gcd == 1:
                        # Number does not share a common factor >1 with last_removed
                        valid_moves.append(i)
        return valid_moves
