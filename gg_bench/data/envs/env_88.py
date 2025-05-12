import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action space: 0 => Add 1, 1 => Multiply by 2
        self.action_space = spaces.Discrete(2)

        # Define observation space:
        # Observation consists of [Current Number, Last Operation Player 1,
        # Last Operation Player 2, Current Player]
        # Last Operations: 0 => None, 1 => Add 1, 2 => Multiply by 2
        # Current Player: 1 or 2
        # Setting reasonable bounds for the current number
        self.target_number = 23  # Default Target Number
        max_current_number = 1000  # Maximum possible current number

        self.observation_space = spaces.Box(
            low=np.array([1, 0, 0, 1], dtype=np.int32),
            high=np.array([max_current_number, 2, 2, 2], dtype=np.int32),
            dtype=np.int32,
        )

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.last_operation = {1: 0, 2: 0}  # 0 => None, 1 => Add 1, 2 => Multiply by 2
        self.current_player = 1
        self.done = False

        # Create observation
        observation = np.array(
            [
                self.current_number,
                self.last_operation[1],
                self.last_operation[2],
                self.current_player,
            ],
            dtype=np.int32,
        )

        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10
            return self.get_observation(), reward, True, False, {}

        # Apply the action
        if action == 0:
            # Add 1
            self.current_number += 1
            operation = 1
        elif action == 1:
            # Multiply by 2
            self.current_number *= 2
            operation = 2

        # Check for win/loss
        if self.current_number == self.target_number:
            # Current player wins
            self.done = True
            reward = 1
            return self.get_observation(), reward, True, False, {}
        elif self.current_number > self.target_number:
            # Current player loses
            self.done = True
            reward = -10
            return self.get_observation(), reward, True, False, {}

        # Update last operation for current player
        self.last_operation[self.current_player] = operation

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        return self.get_observation(), 0, False, False, {}

    def get_observation(self):
        observation = np.array(
            [
                self.current_number,
                self.last_operation[1],
                self.last_operation[2],
                self.current_player,
            ],
            dtype=np.int32,
        )
        return observation

    def render(self):
        return (
            f"Current Number: {self.current_number}\n"
            f"Last Operation Player 1: {self.operation_str(self.last_operation[1])}\n"
            f"Last Operation Player 2: {self.operation_str(self.last_operation[2])}\n"
            f"Current Player: Player {self.current_player}"
        )

    def operation_str(self, operation):
        if operation == 0:
            return "None"
        elif operation == 1:
            return "Add 1"
        elif operation == 2:
            return "Multiply by 2"

    def valid_moves(self):
        # Cannot choose the same operation as player's previous turn
        last_op = self.last_operation[self.current_player]
        if last_op == 0:
            # First turn, can choose any action
            return [0, 1]
        elif last_op == 1:
            # Last action was Add 1, cannot choose Add 1 again
            return [1]  # 1 => Multiply by 2
        elif last_op == 2:
            # Last action was Multiply by 2, cannot choose Multiply by 2 again
            return [0]  # 0 => Add 1
