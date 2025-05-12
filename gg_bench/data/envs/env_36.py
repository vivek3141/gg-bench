import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 4 operations and 9 numbers, so total actions = 36
        self.action_space = spaces.Discrete(36)

        # Observation space:
        # - Running total: integer between -1000 and 1000
        # - Player 0's available numbers: 9 binary values
        # - Player 1's available numbers: 9 binary values
        # - Current player: 0 or 1
        self.observation_space = spaces.Box(
            low=np.array([-1000] + [0] * 19),
            high=np.array([1000] + [1] * 19),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.running_total = 0
        # Both players have numbers 1-9 available (represented as array of 1s)
        self.player_numbers = {
            0: np.ones(9, dtype=np.int32),
            1: np.ones(9, dtype=np.int32),
        }
        self.current_player = 0  # Player 0 starts
        self.done = False
        observation = self._get_observation()
        info = {}
        return observation, info  # Return observation and info

    def _get_observation(self):
        # Return observation as a numpy array:
        # [running_total, player 0 numbers (9), player 1 numbers (9), current_player]
        observation = np.array(
            [self.running_total]
            + self.player_numbers[0].tolist()
            + self.player_numbers[1].tolist()
            + [self.current_player],
            dtype=np.int32,
        )
        return observation

    def step(self, action):
        if self.done:
            observation = self._get_observation()
            reward = 0
            info = {}
            return observation, reward, self.done, False, info

        op_index = action // 9
        number = (action % 9) + 1  # Numbers 1 to 9
        operations = ["+", "-", "*", "/"]
        operation = operations[op_index]

        # Check if number is available
        if self.player_numbers[self.current_player][number - 1] == 0:
            reward = -10
            info = {"reason": "Number not available"}
            self.current_player = 1 - self.current_player  # Switch player
            observation = self._get_observation()
            return observation, reward, False, False, info

        # Perform operation
        prev_total = self.running_total
        if operation == "+":
            self.running_total += number
        elif operation == "-":
            self.running_total -= number
        elif operation == "*":
            self.running_total *= number
        elif operation == "/":
            # Integer division
            if number == 0:
                # Division by zero; but number can't be zero
                reward = -10
                info = {"reason": "Division by zero"}
                self.current_player = 1 - self.current_player
                observation = self._get_observation()
                return observation, reward, False, False, info
            else:
                self.running_total = int(self.running_total / number)

        # Mark number as used
        self.player_numbers[self.current_player][number - 1] = 0

        # Check for victory
        if self.running_total >= 30:
            reward = 1
            self.done = True
            info = {}
            observation = self._get_observation()
            return observation, reward, True, False, info

        # Switch player
        self.current_player = 1 - self.current_player

        # Check if the next player has valid moves
        if not self.has_valid_moves(self.current_player):
            # Check if current player has valid moves
            if not self.has_valid_moves(1 - self.current_player):
                # Neither player has valid moves, determine winner
                self.done = True
                if self.running_total >= 30:
                    reward = 1
                    info = {"reason": "Running total exceeds target"}
                else:
                    # Last player to make valid move wins
                    reward = 1
                    info = {"reason": "Stalemate, last player wins"}
                observation = self._get_observation()
                return observation, reward, True, False, info

        # Game continues
        reward = 0
        observation = self._get_observation()
        return observation, reward, False, False, {}

    def has_valid_moves(self, player_id):
        # Returns True if player has any valid moves
        return np.any(self.player_numbers[player_id] == 1)

    def render(self):
        rendering = ""
        rendering += f"Running Total: {self.running_total}\n"
        rendering += f"Current Player: Player {self.current_player}\n"
        rendering += f"Player 0 Available Numbers: {self._numbers_to_string(self.player_numbers[0])}\n"
        rendering += f"Player 1 Available Numbers: {self._numbers_to_string(self.player_numbers[1])}\n"
        return rendering

    def _numbers_to_string(self, numbers_array):
        numbers = []
        for i, available in enumerate(numbers_array):
            if available == 1:
                numbers.append(str(i + 1))
        return ", ".join(numbers) if numbers else "None"

    def valid_moves(self):
        # Returns a list of valid action indices for the current player
        valid_actions = []
        player_numbers = self.player_numbers[self.current_player]
        for op_index in range(4):
            for number_index in range(9):
                if player_numbers[number_index] == 1:
                    action = op_index * 9 + number_index
                    valid_actions.append(action)
        return valid_actions
