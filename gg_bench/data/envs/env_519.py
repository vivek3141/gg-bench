import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers from 1 to 20, represented by indices 0 to 19
        self.action_space = spaces.Discrete(20)

        # Define observation space:
        # - first 20 entries indicate availability of numbers 1-20 (1 for available, 0 for removed)
        # - 21st entry indicates the last number removed by the opponent (0 if no number has been removed yet)
        self.observation_space = spaces.Box(low=0, high=20, shape=(21,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            20, dtype=np.int32
        )  # Numbers 1 to 20 are all available
        self.last_opponent_action = 0  # No number has been removed yet
        self.current_player = 0  # Player 1 starts (can be 0 or 1)
        self.done = False  # Game is not over
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}  # Game is over

        number = action + 1  # Map action index to the actual number (1 to 20)

        # Check if the selected number is available
        if self.available_numbers[action] == 0:
            # Invalid move: number already removed
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Get list of valid moves for the current player
        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid move according to the game rules
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move: remove the number from the available numbers
        self.available_numbers[action] = 0
        self.last_opponent_action = (
            number  # Update last number removed (for the opponent's turn)
        )

        # Switch to the next player
        self.current_player = 1 - self.current_player

        # Check if the next player has any valid moves
        if len(self.valid_moves()) == 0:
            # The previous player wins since the current player cannot move
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Game continues
        return self._get_observation(), 0, False, False, {}

    def _get_observation(self):
        # Concatenate available numbers and the last opponent action into a single observation array
        observation = np.append(self.available_numbers, self.last_opponent_action)
        return observation

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        valid_moves = []
        if self.last_opponent_action == 0:
            # First move: any available number is valid
            valid_moves = [i for i in range(20) if self.available_numbers[i] == 1]
        else:
            # Subsequent moves: number must be a factor or multiple of the last opponent's number
            last_number = self.last_opponent_action
            for i in range(20):
                if self.available_numbers[i] == 1:
                    number = i + 1
                    if number % last_number == 0 or last_number % number == 0:
                        valid_moves.append(i)
        return valid_moves

    def render(self):
        # Create a string representation of the current game state
        available_numbers_str = "Numbers available: " + " ".join(
            [str(i + 1) for i in range(20) if self.available_numbers[i] == 1]
        )
        last_number = self.last_opponent_action
        current_player = self.current_player + 1  # For display purposes (Player 1 or 2)
        state_str = (
            f"{available_numbers_str}\n"
            f"Player {current_player}'s turn.\n"
            f"Last number removed by opponent: {last_number}\n"
        )
        return state_str

    def valid_moves_list(self):
        # Return the actual numbers corresponding to valid moves (for informational purposes)
        return [i + 1 for i in self.valid_moves()]
