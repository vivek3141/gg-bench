import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Add 1, 1 - Subtract 1, 2 - Divide by 2, 3 - Multiply by 3 and Add 1
        self.action_space = spaces.Discrete(4)

        # Observation space: [shared_number, last_operation]
        # shared_number: integer between 1 and 100
        # last_operation: -1 (no previous operation) or 0 to 3 corresponding to actions
        self.observation_space = spaces.Box(
            low=np.array([1, -1]), high=np.array([100, 3]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = 12  # Starting number is 12
        self.current_player = 1  # Player 1 starts
        self.last_operation_player1 = -1  # No previous operation
        self.last_operation_player2 = -1  # No previous operation
        self.done = False

        observation = np.array([self.shared_number, -1], dtype=np.int32)
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            observation = np.array(
                [self.shared_number, self.get_last_operation()], dtype=np.int32
            )
            return observation, reward, self.done, False, {}

        # Apply the action
        self.apply_action(action)

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Check if the next player has any valid moves
        if not self.valid_moves():
            # Current player wins (opponent has no valid moves)
            self.done = True
            reward = 1
        else:
            self.done = False
            reward = 0

        observation = np.array(
            [self.shared_number, self.get_last_operation()], dtype=np.int32
        )
        return observation, reward, self.done, False, {}

    def render(self):
        return (
            f"Current number: {self.shared_number}, Player {self.current_player}'s turn"
        )

    def valid_moves(self):
        # Returns a list of valid action indices for the current player
        valid_actions = []

        # Determine the properties of the current shared number
        is_prime = self.is_prime(self.shared_number)
        is_even = self.shared_number % 2 == 0
        is_odd = self.shared_number % 2 == 1

        # Get the last operation of the current player
        last_operation = self.get_last_operation()

        # Define the operations
        operations = {
            0: "add",  # Add 1
            1: "subtract",  # Subtract 1
            2: "divide",  # Divide by 2
            3: "multiply",  # Multiply by 3 and Add 1
        }

        # Check each possible action
        for action in range(4):
            if action == last_operation:
                continue  # Cannot repeat the same operation

            # Check if the operation is valid based on the shared number's properties
            result = None
            if action == 0 and is_prime:
                result = self.shared_number + 1
            elif action == 1 and is_prime:
                result = self.shared_number - 1
            elif action == 2 and is_even:
                result = self.shared_number // 2
            elif action == 3 and is_odd:
                result = self.shared_number * 3 + 1

            if result is not None and 1 <= result <= 100 and float(result).is_integer():
                valid_actions.append(action)

        return valid_actions

    def get_last_operation(self):
        if self.current_player == 1:
            return self.last_operation_player1
        else:
            return self.last_operation_player2

    def apply_action(self, action):
        # Apply the selected action to the shared number
        if action == 0:  # Add 1
            self.shared_number += 1
        elif action == 1:  # Subtract 1
            self.shared_number -= 1
        elif action == 2:  # Divide by 2
            self.shared_number = self.shared_number // 2
        elif action == 3:  # Multiply by 3 and Add 1
            self.shared_number = self.shared_number * 3 + 1

        # Update the last operation for the current player
        if self.current_player == 1:
            self.last_operation_player1 = action
        else:
            self.last_operation_player2 = action

    @staticmethod
    def is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while (i * i) <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
