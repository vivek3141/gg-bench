import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Numbers from 2 to 20 inclusive
        self.numbers = np.arange(2, 21)
        self.n_numbers = len(self.numbers)

        # Define action and observation space
        # Actions correspond to indices of self.numbers (0 to 18)
        self.action_space = spaces.Discrete(self.n_numbers)
        # Observation space includes:
        # - Available numbers (1 if available, 0 if not): shape (19,)
        # - Last number selected (0 if none): shape (1,)
        self.observation_space = spaces.Box(
            low=0, high=20, shape=(self.n_numbers + 1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize available numbers and sequence
        self.available_numbers = np.ones(self.n_numbers, dtype=np.int32)
        self.sequence = []
        self.last_number = 0  # No number selected yet

        # Player 1 starts (1 for Player 1, -1 for Player 2)
        self.current_player = 1
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action to selected number
        if action < 0 or action >= self.n_numbers:
            # Invalid action index
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        selected_number = self.numbers[action]

        # Check if number is available
        if self.available_numbers[action] == 0:
            # Number not available
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Validate the move
        valid = False
        if len(self.sequence) == 0:
            # First move by Player 1
            valid = True
        else:
            last_num = self.sequence[-1]
            if (last_num % selected_number == 0) or (selected_number % last_num == 0):
                valid = True

        if not valid:
            # Invalid move
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Valid move, update state
        self.available_numbers[action] = 0
        self.sequence.append(selected_number)
        self.last_number = selected_number

        # Check if the next player has valid moves
        self.current_player *= -1  # Switch player
        valid_moves_next_player = self.valid_moves()

        if not valid_moves_next_player:
            # Next player cannot move, current player wins
            reward = 1  # Current player wins
            self.done = True
            # Switch back to the winning player for correct reward assignment
            self.current_player *= -1
            return self._get_observation(), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            return self._get_observation(), reward, False, False, {}

    def render(self):
        available_nums = self.numbers[self.available_numbers == 1]
        sequence_str = "Sequence: " + " -> ".join(map(str, self.sequence))
        available_str = "Available Numbers: " + ", ".join(map(str, available_nums))
        player_str = (
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )
        return f"{sequence_str}\n{available_str}\n{player_str}\n"

    def valid_moves(self):
        if not self.sequence:
            # First move, any number is valid
            valid_indices = np.where(self.available_numbers == 1)[0]
            return valid_indices.tolist()
        else:
            last_num = self.sequence[-1]
            valid_indices = []
            for idx in np.where(self.available_numbers == 1)[0]:
                num = self.numbers[idx]
                if last_num % num == 0 or num % last_num == 0:
                    valid_indices.append(idx)
            return valid_indices

    def _get_observation(self):
        # Observation includes available numbers and last selected number
        observation = np.zeros(self.n_numbers + 1, dtype=np.int32)
        observation[: self.n_numbers] = self.available_numbers
        observation[-1] = self.last_number
        return observation
