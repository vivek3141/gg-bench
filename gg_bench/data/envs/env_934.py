import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: numbers from 1 to 9 (inclusive)
        # Actions are 0-8 corresponding to numbers 1-9
        self.action_space = spaces.Discrete(9)

        # Define observation space
        # The stack can have a maximum size; we'll set it to 100 for practical purposes
        # Empty positions will be represented with 0
        self.max_stack_size = 100
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.max_stack_size,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the stack and game variables
        self.stack = np.zeros(self.max_stack_size, dtype=np.int32)
        self.stack_top = -1  # No numbers on the stack yet
        self.current_player = 1  # Player 1 starts
        self.done = False

        return self._get_observation(), {}  # Return observation and info

    def _get_observation(self):
        return self.stack.copy()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Convert action into the number (1-9)
        number = action + 1

        # Place the number on top of the stack
        self.stack_top += 1
        if self.stack_top >= self.max_stack_size:
            raise Exception("Stack overflow: Maximum stack size exceeded.")

        self.stack[self.stack_top] = number

        # Initialize reward
        reward = 0

        # Check for win or lose condition
        if self.stack_top >= 1:
            top = self.stack[self.stack_top]
            below_top = self.stack[self.stack_top - 1]
            sum_top_two = top + below_top

            if sum_top_two == 10:
                self.done = True
                reward = 1  # Current player wins
                return self._get_observation(), reward, True, False, {}
            elif sum_top_two > 10:
                self.done = True
                reward = -10  # Current player loses
                return self._get_observation(), reward, True, False, {}
            # If sum_top_two < 10, the game continues
        else:
            # First move: cannot win or lose
            pass  # reward remains 0

        # Switch to the next player
        self.current_player = 1 if self.current_player == 2 else 2

        return self._get_observation(), reward, False, False, {}

    def render(self):
        stack_str = "Stack from bottom to top:\n"
        for i in range(self.stack_top + 1):
            stack_str += str(self.stack[i]) + " "
        return stack_str.strip()

    def valid_moves(self):
        # All numbers from 1 to 9 are always available
        # Valid moves are those that do not cause the player to lose immediately
        valid = []

        for action in range(9):
            number = action + 1
            # Simulate placing the number on the stack
            if self.stack_top >= 0:
                top = number
                below_top = self.stack[self.stack_top]
                sum_top_two = top + below_top

                if sum_top_two <= 10:
                    valid.append(action)
            else:
                # On the first move, all actions are valid
                valid.append(action)
        return valid
