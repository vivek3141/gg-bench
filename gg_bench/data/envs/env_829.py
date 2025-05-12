import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Actions correspond to numbers 1-9
        # Observation space includes availability of numbers 1-9 and last opponent's number chosen
        self.observation_space = spaces.Box(low=0, high=9, shape=(10,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            9, dtype=np.int8
        )  # Numbers 1-9 are all available at the start
        self.last_number_chosen = 0  # No last number chosen at the start
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            # If game is over, return current state
            observation = self._get_observation()
            return observation, 0, True, False, {}

        number_chosen = action + 1  # Convert action index to number (1-9)

        if self.available_numbers[action] == 0:
            # Number already crossed off; invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Check if move is valid
        if self.last_number_chosen == 0:
            # First move; any number is valid
            valid_move = True
        else:
            # Move must be a divisor or multiple of the last number chosen by opponent
            if (
                self.last_number_chosen % number_chosen == 0
                or number_chosen % self.last_number_chosen == 0
            ):
                valid_move = True
            else:
                valid_move = False

        if not valid_move:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Valid move; update game state
        self.available_numbers[action] = 0  # Cross off the chosen number
        self.last_number_chosen = number_chosen  # Update last number chosen

        # Switch to the next player
        self.current_player *= -1

        # Check if the next player has any valid moves
        if len(self.valid_moves()) == 0:
            # No valid moves left for the next player; current player wins
            self.done = True
            reward = 1  # Reward for winning
            observation = self._get_observation()
            return observation, reward, True, False, {}
        else:
            # Game continues
            reward = 0  # No immediate reward
            observation = self._get_observation()
            return observation, reward, False, False, {}

    def render(self):
        numbers_state = [
            f" {i+1} " if self.available_numbers[i] == 1 else " X " for i in range(9)
        ]
        output = "\nCurrent Board State:\n"
        output += "---------------------\n"
        output += f"Available Numbers: {', '.join(numbers_state)}\n"
        output += f"Last Number Chosen by Opponent: {self.last_number_chosen}\n"
        output += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        output += "---------------------\n"
        return output

    def valid_moves(self):
        valid_actions = []
        for action in range(9):
            if self.available_numbers[action] == 1:
                number = action + 1
                if self.last_number_chosen == 0:
                    # First move; all available numbers are valid
                    valid_actions.append(action)
                else:
                    # Check divisibility conditions
                    if (
                        self.last_number_chosen % number == 0
                        or number % self.last_number_chosen == 0
                    ):
                        valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Observation includes available numbers and last number chosen by opponent
        observation = np.append(self.available_numbers, self.last_number_chosen)
        return observation
