import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define constants
        self.MAX_STACK_SIZE = 30  # Maximum stack size

        # Define action space:
        # Actions 0-8: Push numbers 1-9
        # Actions 9-11: Operate with '+', '-', '*'
        self.action_space = spaces.Discrete(12)

        # Define observation space: fixed-size stack with padding
        # Stack elements can be any integer value
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.MAX_STACK_SIZE,), dtype=np.int32
        )

        # Initialize environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the stack as a numpy array filled with zeros
        self.stack = np.zeros(self.MAX_STACK_SIZE, dtype=np.int32)
        self.stack_pointer = 0  # Points to the next free position in the stack
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array(self.stack, copy=True), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, return the current state
            return (
                np.array(self.stack, copy=True),
                0,
                True,
                False,
                {},
            )

        reward = 0  # Default reward
        info = {}

        # Check if the action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return np.array(self.stack, copy=True), reward, True, False, info

        if action >= 0 and action <= 8:
            # Push action
            number_to_push = action + 1  # Map action to number 1-9
            if self.stack_pointer >= self.MAX_STACK_SIZE:
                # Stack overflow, invalid move
                self.done = True
                reward = -10  # Penalty for invalid move
                return np.array(self.stack, copy=True), reward, True, False, info
            self.stack[self.stack_pointer] = number_to_push
            self.stack_pointer += 1
        elif action >= 9 and action <= 11:
            # Operate action
            if self.stack_pointer < 2:
                # Not enough operands to operate, invalid move
                self.done = True
                reward = -10  # Penalty for invalid move
                return np.array(self.stack, copy=True), reward, True, False, info
            # Pop the operands
            self.stack_pointer -= 1
            second_operand = self.stack[self.stack_pointer]
            self.stack[self.stack_pointer] = 0  # Clear the value
            self.stack_pointer -= 1
            first_operand = self.stack[self.stack_pointer]
            self.stack[self.stack_pointer] = 0  # Clear the value

            # Choose the operation
            if action == 9:
                # Addition
                result = first_operand + second_operand
            elif action == 10:
                # Subtraction
                result = first_operand - second_operand
            elif action == 11:
                # Multiplication
                result = first_operand * second_operand
            else:
                # Invalid operation, should not happen
                self.done = True
                reward = -10  # Penalty for invalid move
                return np.array(self.stack, copy=True), reward, True, False, info

            # Push the result back onto the stack
            self.stack[self.stack_pointer] = result
            self.stack_pointer += 1
        else:
            # Invalid action index, should not happen
            self.done = True
            reward = -10  # Penalty for invalid move
            return np.array(self.stack, copy=True), reward, True, False, info

        # Check victory condition
        if self.stack_pointer == 1 and self.stack[0] == 24:
            # Current player wins
            self.done = True
            reward = 1  # Reward for winning
            return np.array(self.stack, copy=True), reward, True, False, info

        # Switch to the other player for self-play
        self.current_player *= -1

        # Continue the game
        return np.array(self.stack, copy=True), reward, False, False, info

    def render(self):
        # Generate string representation of the stack
        stack_str = "Current Stack (top -> bottom):\n"
        if self.stack_pointer == 0:
            stack_str += "Empty Stack"
        else:
            for i in range(self.stack_pointer - 1, -1, -1):
                stack_str += f"{self.stack[i]}\n"
        return stack_str

    def valid_moves(self):
        # Returns a list of valid action indices
        valid_actions = []

        # Push actions are always valid unless the stack is full
        if self.stack_pointer < self.MAX_STACK_SIZE:
            valid_actions.extend(
                range(0, 9)
            )  # Actions 0-8 correspond to push numbers 1-9

        # Operate actions are valid only if there are at least two numbers on the stack
        if self.stack_pointer >= 2:
            valid_actions.extend(
                range(9, 12)
            )  # Actions 9-11 correspond to '+', '-', '*'

        return valid_actions
