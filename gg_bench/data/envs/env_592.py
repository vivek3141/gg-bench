import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers from 2 to 50 inclusive
        self.action_space = spaces.Discrete(49)  # Actions correspond to numbers 2 to 50

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "last_number": spaces.Box(low=0, high=50, shape=(1,), dtype=np.int32),
                "available_numbers": spaces.Box(
                    low=0, high=1, shape=(49,), dtype=np.int8
                ),
            }
        )

        # Initialize state variables
        self.available_numbers = None
        self.last_number = None
        self.current_player = None
        self.done = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            49, dtype=np.int8
        )  # Numbers from 2 to 50 are available
        self.last_number = 0  # No last number chosen
        self.current_player = 1  # Start with Player 1
        self.done = False

        obs = {
            "last_number": np.array([self.last_number], dtype=np.int32),
            "available_numbers": self.available_numbers.copy(),
        }
        return obs, {}  # Return observation and info

    def step(self, action):
        # Check if action is valid index
        if not self.action_space.contains(action):
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}  # Invalid action index

        number = action + 2  # Map action index to number between 2 and 50

        # Check if number is available
        if self.available_numbers[action] != 1:
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}  # Number not available

        # Check if move is valid
        if self.last_number == 0:
            # First move, any number is valid
            valid_move = True
        else:
            if self.last_number % number == 0 or number % self.last_number == 0:
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}  # Invalid move

        # Valid move
        # Remove number from available_numbers
        self.available_numbers[action] = 0
        self.last_number = number

        # Check if opponent has any valid moves
        opponent_valid_moves = self._get_valid_moves(self.last_number)
        if len(opponent_valid_moves) == 0:
            # Opponent cannot move; current player wins
            reward = 1
            self.done = True
            return self._get_obs(), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            self.done = False
            # Switch current_player
            self.current_player = 2 if self.current_player == 1 else 1
            return self._get_obs(), reward, False, False, {}

    def render(self):
        s = f"Player {self.current_player}'s turn.\n"
        if self.last_number == 0:
            s += "No last number selected.\n"
        else:
            s += f"Last number selected: {self.last_number}\n"
        available_numbers = [i + 2 for i in range(49) if self.available_numbers[i] == 1]
        s += f"Available numbers: {available_numbers}\n"
        return s

    def valid_moves(self):
        return self._get_valid_moves(self.last_number)

    def _get_valid_moves(self, last_number):
        # Returns a list of action indices that are valid moves for the given last_number
        if last_number == 0:
            # First move, all available numbers
            valid_moves = [i for i in range(49) if self.available_numbers[i] == 1]
        else:
            valid_moves = []
            for i in range(49):
                if self.available_numbers[i] == 1:
                    number = i + 2
                    if last_number % number == 0 or number % last_number == 0:
                        valid_moves.append(i)
        return valid_moves

    def _get_obs(self):
        return {
            "last_number": np.array([self.last_number], dtype=np.int32),
            "available_numbers": self.available_numbers.copy(),
        }
