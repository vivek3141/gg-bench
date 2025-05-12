import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools
import math


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 18 actions (numbers 1-9 with option to attempt equation)
        # Actions 0-17: (number 1-9, attempt_equation True/False)
        self.action_space = spaces.Discrete(18)

        # Observation space: [-1, 0, 1] for numbers 1-9
        # -1: Number collected by opponent
        # 0: Number still available
        # 1: Number collected by current player
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize variables
        self.available_numbers = set(range(1, 10))  # Numbers 1-9
        self.player_numbers = {1: [], -1: []}
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.last_action = None
        self.info = {}
        # Create initial observation
        self.observation = np.zeros(9, dtype=np.int8)
        return self.observation.copy(), self.info  # Return observation and info

    def step(self, action):
        if self.done:
            return self.observation.copy(), 0, True, False, self.info

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10
            return self.observation.copy(), reward, True, False, self.info

        # Map action to number and whether to attempt equation
        number = (action // 2) + 1  # Map action to number 1-9
        attempt_equation = action % 2 == 1

        # Check if the number is available
        if number not in self.available_numbers:
            # Invalid move
            self.done = True
            reward = -10
            return self.observation.copy(), reward, True, False, self.info

        # Update the state
        self.available_numbers.remove(number)
        self.player_numbers[self.current_player].append(number)
        self.player_numbers[self.current_player].sort()
        index = number - 1
        self.observation[index] = 1  # Mark number as collected by current player

        # Remove number from opponent's observation
        opponent = -self.current_player
        if number in self.player_numbers[opponent]:
            self.player_numbers[opponent].remove(number)
            self.observation[index] = 1
        else:
            self.observation[index] = 1

        reward = 0
        terminated = False

        if attempt_equation:
            # Attempt to form a valid equation
            player_nums = self.player_numbers[self.current_player]
            if self.can_form_valid_equation(player_nums):
                # Player wins
                reward = 1
                self.done = True
                terminated = True
                return self.observation.copy(), reward, True, False, self.info
            else:
                # Invalid equation
                reward = -10
                self.done = True
                terminated = True
                return self.observation.copy(), reward, True, False, self.info

        # Check for sudden death condition
        if len(self.available_numbers) == 0:
            # Sudden death mode
            # Players take turns attempting to form a valid equation
            if self.can_form_valid_equation(self.player_numbers[self.current_player]):
                reward = 1
                self.done = True
                return self.observation.copy(), reward, True, False, self.info
            else:
                # Switch player for next attempt
                self.current_player *= -1
                if self.can_form_valid_equation(
                    self.player_numbers[self.current_player]
                ):
                    reward = -1  # Opponent can win in next turn
                    self.done = True
                    return self.observation.copy(), reward, True, False, self.info
                else:
                    # Tie-breaker: highest sum wins
                    sum_current = sum(self.player_numbers[self.current_player])
                    sum_opponent = sum(self.player_numbers[-self.current_player])
                    if sum_current > sum_opponent:
                        reward = 1
                    elif sum_current < sum_opponent:
                        reward = -1
                    else:
                        reward = 0
                    self.done = True
                    return self.observation.copy(), reward, True, False, self.info
        else:
            # Switch to next player
            self.current_player *= -1
            self.observation *= -1  # Flip observation for the next player

        return self.observation.copy(), reward, False, False, self.info

    def render(self):
        s = "Available Numbers: " + str(sorted(self.available_numbers)) + "\n"
        s += "Player {}'s Numbers: {}\n".format(
            1 if self.current_player == 1 else 2,
            self.player_numbers[self.current_player],
        )
        s += "Player {}'s Numbers: {}\n".format(
            2 if self.current_player == 1 else 1,
            self.player_numbers[-self.current_player],
        )
        return s

    def valid_moves(self):
        valid_actions = []
        for num in self.available_numbers:
            idx = (num - 1) * 2
            valid_actions.append(idx)  # Select number without equation attempt
            valid_actions.append(idx + 1)  # Select number with equation attempt
        return valid_actions

    def can_form_valid_equation(self, numbers):
        # Check if any valid equation can be formed from the numbers
        # using basic arithmetic operators and parentheses
        # Return True if at least one valid equation exists, else False

        if len(numbers) < 3:
            return False  # Need at least three numbers to form an equation

        nums = numbers.copy()
        operators_list = ["+", "-", "*", "/"]
        nums_permutations = set(itertools.permutations(nums))

        for nums_perm in nums_permutations:
            nums_perm = list(nums_perm)
            n = len(nums_perm)
            # Split numbers into left and right sides
            for i in range(1, n):  # Split point
                left_nums = nums_perm[:i]
                right_nums = nums_perm[i:]
                left_exprs = self.generate_expressions(left_nums)
                right_exprs = self.generate_expressions(right_nums)
                for left_expr in left_exprs:
                    for right_expr in right_exprs:
                        try:
                            if self.safe_eval(left_expr) == self.safe_eval(right_expr):
                                # Valid equation found
                                # print(f"Valid equation: {left_expr} = {right_expr}")
                                return True
                        except ZeroDivisionError:
                            continue
                        except Exception:
                            continue
        return False

    def generate_expressions(self, numbers):
        # Generate all possible expressions from numbers using operators and parentheses
        if len(numbers) == 1:
            return [str(numbers[0])]

        expressions = []
        operators = ["+", "-", "*", "/"]
        for i in range(1, len(numbers)):
            left_parts = self.generate_expressions(numbers[:i])
            right_parts = self.generate_expressions(numbers[i:])
            for left in left_parts:
                for right in right_parts:
                    for op in operators:
                        expr = f"({left}{op}{right})"
                        expressions.append(expr)
        return expressions

    def safe_eval(self, expr):
        # Evaluate expression safely
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        code = compile(expr, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of '{name}' not allowed")
        return eval(code, {"__builtins__": {}}, allowed_names)
