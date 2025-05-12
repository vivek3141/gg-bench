import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(9) for numbers 1-9
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # - Indices 0-8: Availability of numbers 1-9 (1 if available, 0 if not)
        # - Index 9: Last opponent's selected number (0 if none)
        self.observation_space = spaces.Box(low=0, high=9, shape=(10,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(9, dtype=np.int32)  # Numbers 1-9 are available
        self.last_opponent_number = 0  # No last opponent number at the start
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}  # Game is over

        number_chosen = action + 1  # Map action (0-8) to number (1-9)

        # Check if the number is available
        if self.available_numbers[action] == 0:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Check if the move is valid
        if self.last_opponent_number == 0:
            # First move; any available number is valid
            valid_move = True
        else:
            # Valid if number is a factor or multiple of last opponent's number
            if (
                number_chosen % self.last_opponent_number == 0
                or self.last_opponent_number % number_chosen == 0
            ):
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Valid move; update game state
        self.available_numbers[action] = 0  # Remove the number from available numbers
        self.last_opponent_number = number_chosen  # Update last opponent number

        # Check if the opponent has valid moves
        opponent_valid_moves = self._get_valid_moves()
        if len(opponent_valid_moves) == 0:
            # Opponent cannot move; current player wins
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}
        else:
            # Switch to the next player
            self.current_player *= -1
            reward = 0
            return self._get_observation(), reward, False, False, {}

    def render(self):
        # Visual representation of the game state
        available_nums = [
            str(i + 1) for i in range(9) if self.available_numbers[i] == 1
        ]
        taken_nums = [str(i + 1) for i in range(9) if self.available_numbers[i] == 0]
        state_str = "Game State:\n"
        state_str += f"Available Numbers: {available_nums}\n"
        state_str += f"Last Opponent's Number: {self.last_opponent_number}\n"
        return state_str

    def valid_moves(self):
        # Return a list of valid moves as indices of the action_space
        return self._get_valid_moves()

    def _get_valid_moves(self):
        # Helper method to get valid moves based on game rules
        valid_moves = []
        for i in range(9):
            if self.available_numbers[i] == 1:
                number = i + 1  # Number corresponding to the action
                if self.last_opponent_number == 0:
                    # First move; any available number is valid
                    valid_moves.append(i)
                else:
                    if (
                        number % self.last_opponent_number == 0
                        or self.last_opponent_number % number == 0
                    ):
                        valid_moves.append(i)
        return valid_moves

    def _get_observation(self):
        # Build the observation array
        obs = np.zeros(10, dtype=np.int32)
        obs[:9] = self.available_numbers
        obs[9] = self.last_opponent_number
        return obs
