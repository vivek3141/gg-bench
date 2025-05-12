import numpy as np
import gymnasium as gym
from gymnasium import spaces

MAX_SEQUENCE_LENGTH = 20


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are numbers 1 to 9, represented by indices 0 to 8
        self.action_space = spaces.Discrete(9)
        # Observation space is the sequence with a fixed maximum length
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(MAX_SEQUENCE_LENGTH,), dtype=np.int8
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the sequence and other variables
        self.sequence = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.int8)
        self.current_index = 0  # Next empty position in the sequence
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.sequence.copy(), {}  # Return observation and info

    def step(self, action):
        # Check for invalid action (e.g., game already over)
        if self.done:
            return self.sequence.copy(), -10, True, False, {}

        if action < 0 or action >= self.action_space.n:
            self.done = True
            return self.sequence.copy(), -10, True, False, {}

        # Map action to number (1-9)
        number = action + 1  # Actions 0-8 correspond to numbers 1-9

        # Add the number to the sequence if there's space
        if self.current_index >= MAX_SEQUENCE_LENGTH:
            self.done = True
            return self.sequence.copy(), 0, True, False, {}

        self.sequence[self.current_index] = number
        self.current_index += 1

        # Initialize reward and check for winning condition
        reward = 0

        if self.current_index >= 3:
            a = self.sequence[self.current_index - 3]
            b = self.sequence[self.current_index - 2]
            c = self.sequence[self.current_index - 1]

            # Check if the last three numbers form a valid equation
            if self.is_valid_equation(a, b, c):
                self.done = True
                reward = 1
                return self.sequence.copy(), reward, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        return self.sequence.copy(), reward, False, False, {}

    def is_valid_equation(self, a, b, c):
        # Check for valid equations of the forms:
        # a op b == c and a == b op c
        operators = ["+", "-", "*", "/"]

        for op in operators:
            # Check a op b == c
            result1 = self.apply_op(a, b, op)
            if result1 is not None and result1 == c:
                return True

            # Check a == b op c
            result2 = self.apply_op(b, c, op)
            if result2 is not None and a == result2:
                return True
        return False

    def apply_op(self, x, y, op):
        try:
            if op == "+":
                return x + y
            elif op == "-":
                return x - y
            elif op == "*":
                return x * y
            elif op == "/":
                # Check for division by zero and integer division validity
                if y != 0 and x % y == 0:
                    return x // y
                else:
                    return None
            else:
                return None
        except:
            return None

    def render(self):
        # Return a string representation of the current sequence
        sequence_str = "Sequence: " + " ".join(
            str(int(num)) for num in self.sequence[: self.current_index]
        )
        return sequence_str

    def valid_moves(self):
        # Return a list of valid action indices (0-8) if the game isn't over
        if self.done:
            return []
        else:
            return list(range(self.action_space.n))
