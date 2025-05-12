import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(49)  # Actions correspond to numbers 2 to 50
        self.observation_space = spaces.Box(
            low=0, high=50, shape=(50,), dtype=np.float32
        )  # 49 numbers + opponent's last number

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.number_pool = np.ones(
            49, dtype=np.float32
        )  # Numbers 2 to 50 are available
        self.first_move = True  # Indicates if it's the first move of the game
        self.current_player = 1  # Player 1 starts
        self.opponent_last_number = 0  # No opponent's last number at the start
        self.current_player_last_number = 0  # Current player's last selected number
        self.done = False

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, False, {}

        selected_number = action + 2  # Map action to number (2 to 50)
        info = {}

        # Check if selected number is available
        if self.number_pool[action] == 0:
            self.done = True
            reward = -10  # Invalid move
            return self._get_observation(), reward, self.done, False, info

        # Check if move is valid
        if self.first_move:
            valid = True  # Any number is valid on the first move
        else:
            if self.opponent_last_number == 0:
                valid = False
            else:
                # Move is valid if selected number is a divisor or multiple of opponent's last number
                if (
                    selected_number % self.opponent_last_number == 0
                    or self.opponent_last_number % selected_number == 0
                ):
                    valid = True
                else:
                    valid = False

        if not valid:
            self.done = True
            reward = -10  # Invalid move
            return self._get_observation(), reward, self.done, False, info

        # Move is valid, update the game state
        self.number_pool[action] = 0  # Remove the number from the pool
        self.first_move = False
        self.current_player_last_number = selected_number

        # Check if opponent has any valid moves
        opponent_valid_moves = self._get_valid_moves(self.current_player_last_number)
        if not opponent_valid_moves:
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, self.done, False, info
        else:
            # Switch players
            self.opponent_last_number = self.current_player_last_number
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
            reward = 0  # Game continues
            return self._get_observation(), reward, self.done, False, info

    def render(self):
        numbers_available = [i + 2 for i in range(49) if self.number_pool[i] == 1]
        numbers_taken = [i + 2 for i in range(49) if self.number_pool[i] == 0]
        board_str = "Numbers available: {}\n".format(numbers_available)
        board_str += "Numbers taken: {}\n".format(numbers_taken)
        board_str += "Current player: Player {}\n".format(self.current_player)
        if self.opponent_last_number != 0:
            board_str += "Opponent's last number: {}\n".format(
                self.opponent_last_number
            )
        else:
            board_str += "Opponent's last number: None\n"
        return board_str

    def valid_moves(self):
        valid_moves_indices = self._get_valid_moves(self.opponent_last_number)
        return valid_moves_indices

    def _get_observation(self):
        observation = np.concatenate(
            [self.number_pool.copy(), [self.opponent_last_number]]
        )
        return observation.astype(np.float32)

    def _get_valid_moves(self, last_number):
        if self.first_move or last_number == 0:
            # All available numbers are valid on the first move
            valid_moves_indices = [i for i in range(49) if self.number_pool[i] == 1]
        else:
            valid_moves_indices = []
            for i in range(49):
                if self.number_pool[i] == 1:
                    number = i + 2
                    if number % last_number == 0 or last_number % number == 0:
                        valid_moves_indices.append(i)
        return valid_moves_indices
