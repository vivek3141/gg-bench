import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers 1-9 are actions 0-8
        self.action_space = spaces.Discrete(9)

        # Define observation space: the stack with a maximum size
        self.MAX_STACK_SIZE = 20  # You can adjust the maximum size as needed
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.MAX_STACK_SIZE,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.stack = np.zeros(self.MAX_STACK_SIZE, dtype=np.int32)
        self.top_index = 0  # Index to place the next number
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game over flag
        return self.stack.copy(), {}  # Return initial observation and info

    def is_action_valid(self, number):
        if self.stack[0] == 0:
            # If the stack is empty, any number is valid
            return True
        else:
            top_number = self.stack[self.top_index - 1]
            # Check if number is a factor or multiple of the top number
            return number % top_number == 0 or top_number % number == 0

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        valid_actions = []
        for action in range(9):
            number = action + 1  # Map action to number 1-9
            if self.is_action_valid(number):
                valid_actions.append(action)
        return valid_actions

    def step(self, action):
        if self.done:
            # If the game is over, no further actions are allowed
            return self.stack.copy(), -10, True, False, {}

        number = action + 1  # Map action index to number 1-9

        # Check if the action is valid
        if action not in self.valid_moves():
            self.done = True
            return self.stack.copy(), -10, True, False, {}  # Invalid move

        # Place the number on the stack
        if self.top_index < self.MAX_STACK_SIZE:
            self.stack[self.top_index] = number
            self.top_index += 1
        else:
            # Stack overflow, should not happen with reasonable MAX_STACK_SIZE
            self.done = True
            return self.stack.copy(), -10, True, False, {}

        # Check if the opponent has any valid moves
        self.current_player *= -1  # Switch to opponent
        opponent_valid_moves = self.valid_moves()
        if not opponent_valid_moves:
            # Opponent cannot make a move; current player wins
            self.done = True
            reward = 1
            # No need to switch back current_player since game is over
        else:
            # Continue the game
            reward = 0
            self.current_player *= -1  # Switch back to current player

        return (
            self.stack.copy(),
            reward,
            self.done,
            False,
            {},
        )  # Observation, reward, done, truncated, info

    def render(self):
        # Return a string representation of the current stack
        stack_repr = "Current Stack: "
        if self.stack[0] == 0:
            stack_repr += "Empty"
        else:
            stack_numbers = [str(num) for num in self.stack[: self.top_index]]
            stack_repr += "[" + ", ".join(stack_numbers) + "]"
        return stack_repr

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        valid_actions = []
        for action in range(9):
            number = action + 1  # Map action to number 1-9
            if self.is_action_valid(number):
                valid_actions.append(action)
        return valid_actions
