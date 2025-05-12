import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(100)  # Actions correspond to numbers 1-100

        # Observation space:
        # Index 0: last_number_played (0 if no number has been played yet)
        # Indexes 1-100: availability of numbers 1-100 (1 for available, 0 for taken)
        # Index 101: current_player (1 or -1)
        low = np.zeros(102, dtype=np.int32)
        high = np.ones(102, dtype=np.int32)
        high[0] = 100  # last_number_played can be from 0 to 100
        low[101] = -1  # current_player can be -1 or 1
        high[101] = 1
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = np.ones(
            100, dtype=np.int32
        )  # Numbers from 1-100 are available
        self.last_number_played = 0  # No number has been played yet
        self.current_player = 1  # 1 or -1
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if current player has any valid moves
        valid_moves = self._get_valid_moves()
        if len(valid_moves) == 0:
            # Current player cannot move, they lose
            self.done = True
            return self._get_observation(), -1, True, False, {}

        # Convert action to the chosen number
        chosen_number = (
            action + 1
        )  # Since action is from 0 to 99, numbers from 1 to 100

        # Check if number is available
        if self.number_pool[action] == 0:
            # Invalid move: number not available
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if move is valid according to the game rules
        if self.last_number_played == 0:
            # First move, any number is valid
            valid_move = True
        else:
            # First digit of chosen number must match last digit of last_number_played
            first_digit = int(str(chosen_number)[0])
            last_digit = int(str(self.last_number_played)[-1])
            if first_digit == last_digit:
                valid_move = True
            else:
                # Invalid move
                self.done = True
                return self._get_observation(), -10, True, False, {}

        # Valid move
        # Remove the number from the number pool
        self.number_pool[action] = 0
        # Update last number played
        self.last_number_played = chosen_number

        # Switch player
        self.current_player *= -1

        # Check if next player has any valid moves
        opponent_valid_moves = self._get_valid_moves()
        if len(opponent_valid_moves) == 0:
            # Next player cannot move, current player wins
            self.done = True
            # Switch back to current player to indicate the winner
            self.current_player *= -1
            return self._get_observation(), 1, True, False, {}
        else:
            # Continue game
            return self._get_observation(), 0, False, False, {}

    def render(self):
        available_numbers = np.where(self.number_pool == 1)[0] + 1  # Available numbers
        s = f"Current player: {self.current_player}\n"
        s += f"Last number played: {self.last_number_played}\n"
        s += f"Available numbers: {available_numbers}\n"
        return s

    def valid_moves(self):
        return self._get_valid_moves()

    def _get_observation(self):
        observation = np.zeros(102, dtype=np.int32)
        observation[0] = self.last_number_played
        observation[1:101] = self.number_pool
        observation[101] = self.current_player
        return observation

    def _get_valid_moves(self):
        if self.last_number_played == 0:
            # First move, all available numbers are valid
            valid_moves = np.where(self.number_pool == 1)[0]
        else:
            # Need numbers whose first digit matches the last digit of last_number_played
            last_digit = int(str(self.last_number_played)[-1])
            valid_moves = []
            for i in range(100):
                if self.number_pool[i] == 1:
                    number = i + 1  # Map index to number
                    first_digit = int(str(number)[0])
                    if first_digit == last_digit:
                        valid_moves.append(i)
            valid_moves = np.array(valid_moves, dtype=np.int32)
        return valid_moves
