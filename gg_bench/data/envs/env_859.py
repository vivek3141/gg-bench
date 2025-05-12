import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: integers from 0 to 48 corresponding to numbers 2 to 50
        self.action_space = spaces.Discrete(49)
        # Observation space: array of 49 elements representing the status of numbers 2 to 50
        # 0: number is available
        # 1: selected by player 1
        # 2: selected by player 2
        self.observation_space = spaces.Box(low=0, high=2, shape=(49,), dtype=np.int8)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.selected_numbers = (
            {}
        )  # Map of selected numbers to the player who selected them
        self.available_numbers = set(range(2, 51))  # Numbers from 2 to 50
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.observation = np.zeros(49, dtype=np.int8)
        return self.observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.observation, 0, True, False, {}
        number = action + 2  # Map action index to the actual number

        # Check if the action is valid
        if number not in self.available_numbers or not self.is_valid_selection(number):
            # Invalid move
            self.done = True
            reward = -10
            return self.observation, reward, True, False, {}
        else:
            # Valid move
            self.available_numbers.remove(number)
            self.selected_numbers[number] = self.current_player
            self.observation[action] = self.current_player

            # Switch to the next player
            self.current_player = 2 if self.current_player == 1 else 1

            # Check if the next player has any valid moves
            if not self.has_valid_move():
                # Current player wins
                self.done = True
                reward = 1
                return self.observation, reward, True, False, {}
            else:
                # Game continues
                reward = 0
                return self.observation, reward, False, False, {}

    def render(self):
        status = ""
        for i in range(49):
            num = i + 2
            if self.observation[i] == 0:
                status += f"{num} "
            else:
                player = self.observation[i]
                status += f"({num}-P{player}) "
            if (i + 1) % 10 == 0:
                status += "\n"
        return status

    def valid_moves(self):
        valid_moves = []
        for action in range(49):
            number = action + 2
            if number in self.available_numbers and self.is_valid_selection(number):
                valid_moves.append(action)
        return valid_moves

    def is_valid_selection(self, number):
        # A number is valid if it is not a multiple or factor of any previously selected number
        for selected_number in self.selected_numbers:
            if self.is_factor_or_multiple(number, selected_number):
                return False
        return True

    @staticmethod
    def is_factor_or_multiple(num1, num2):
        # Check if num1 is a factor or multiple of num2
        if num1 == num2:
            return True
        if num1 % num2 == 0 or num2 % num1 == 0:
            return True
        return False

    def has_valid_move(self):
        # Check if the current player has any valid moves
        for number in self.available_numbers:
            if self.is_valid_selection(number):
                return True
        return False
