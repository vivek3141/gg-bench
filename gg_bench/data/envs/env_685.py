import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 0-10 corresponding to '1'-'9', '+', '*'
        self.action_space = spaces.Discrete(11)

        # Observation space: sequence of actions (integers), padded with -1
        self.max_length = 50  # Maximum length of the expression
        self.observation_space = spaces.Box(
            low=-1, high=10, shape=(self.max_length,), dtype=np.int32
        )

        # Map actions to characters and vice versa
        self.action_to_char = {
            0: "1",
            1: "2",
            2: "3",
            3: "4",
            4: "5",
            5: "6",
            6: "7",
            7: "8",
            8: "9",
            9: "+",
            10: "*",
        }
        self.char_to_action = {v: k for k, v in self.action_to_char.items()}

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.expression = []
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.total = 0
        self.invalid = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Append action to the expression
        self.expression.append(action)

        # Check if the expression is valid
        if not self._is_valid_expression():
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Evaluate the expression if it ends with a number
        if self._ends_with_number():
            self.total = self._evaluate_expression()
            if self.total == 100:
                self.done = True
                return self._get_observation(), 1, True, False, {}
            elif self.total > 100:
                self.done = True
                return self._get_observation(), -10, True, False, {}
            else:
                # Valid move, game continues
                pass

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1
        return self._get_observation(), 0, False, False, {}

    def render(self):
        expr_str = "".join([self.action_to_char[action] for action in self.expression])
        return expr_str

    def valid_moves(self):
        valid_actions = []

        # If the expression is empty, cannot start with an operator
        if not self.expression:
            valid_actions = list(range(9))  # Numbers only
        else:
            last_action = self.expression[-1]
            last_char = self.action_to_char[last_action]

            if last_char in "+*":
                # After an operator, we can only have a number
                valid_actions = list(range(9))
            else:
                # After a number, we can have a number or operator
                valid_actions = list(range(11))
        return valid_actions

    def _get_observation(self):
        obs = np.full(self.max_length, -1, dtype=np.int32)
        obs[: len(self.expression)] = self.expression
        return obs

    def _is_valid_expression(self):
        expr_chars = [self.action_to_char[action] for action in self.expression]
        if expr_chars[0] in "+*":
            return False  # Cannot start with an operator

        for i in range(len(expr_chars) - 1):
            if expr_chars[i] in "+*" and expr_chars[i + 1] in "+*":
                return False  # Operators cannot be adjacent
        return True

    def _ends_with_number(self):
        if not self.expression:
            return False
        last_action = self.expression[-1]
        last_char = self.action_to_char[last_action]
        return last_char.isdigit()

    def _evaluate_expression(self):
        expr_chars = [self.action_to_char[action] for action in self.expression]
        tokens = []
        number_buffer = ""

        for char in expr_chars:
            if char.isdigit():
                number_buffer += char
            else:
                tokens.append(int(number_buffer))
                tokens.append(char)
                number_buffer = ""
        if number_buffer != "":
            tokens.append(int(number_buffer))

        # Left-to-right evaluation without operator precedence
        result = tokens[0]
        i = 1
        while i < len(tokens):
            operator = tokens[i]
            operand = tokens[i + 1]
            if operator == "+":
                result += operand
            elif operator == "*":
                result *= operand
            i += 2
        return result
