import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.max_eq_length = 20  # Maximum length of the equation space
        self.num_number_tiles = 11  # Digits 0-9 plus extra '1' tile at index 1
        self.num_operator_tiles = 5  # '+', '-', '*', '/', '='
        self.num_total_tiles = (
            self.num_number_tiles + self.num_operator_tiles
        )  # Total tile types
        self.tile_indices = {
            0: "0",
            1: "1",
            2: "1",  # Extra '1' tile
            3: "2",
            4: "3",
            5: "4",
            6: "5",
            7: "6",
            8: "7",
            9: "8",
            10: "9",
            11: "+",
            12: "-",
            13: "*",
            14: "/",
            15: "=",
        }

        # Define action space
        # Actions 0 to 319: Place tile T at position P
        # Action 320: Rearrangement
        # Action 321: End turn
        self.action_space = spaces.Discrete(
            self.num_total_tiles * self.max_eq_length + 2
        )
        self.rearrange_action_code = self.num_total_tiles * self.max_eq_length
        self.end_turn_action_code = self.num_total_tiles * self.max_eq_length + 1

        # Define observation space
        # Observation includes:
        # - Equation space: self.max_eq_length positions, entries from -1 (empty) to 15 (tile indices)
        # - Tile pool counts: 11 counts for number tiles (indices 0 to 10)
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (
                    np.full((self.max_eq_length,), -1),
                    np.zeros(self.num_number_tiles, dtype=int),
                )
            ),
            high=np.concatenate(
                (
                    np.full((self.max_eq_length,), 15),
                    np.array([2] + [1] * 10, dtype=int),
                )
            ),
            shape=(self.max_eq_length + self.num_number_tiles,),
            dtype=np.int8,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the equation space to empty
        self.equation_space = np.full((self.max_eq_length,), -1, dtype=np.int8)
        # Initialize the tile pool counts: indices 0 to 10 correspond to number tiles 0-9 with extra '1' at index 1
        self.tile_pool_counts = np.array([1, 2] + [1] * 9, dtype=np.int8)
        # Initialize game state variables
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.placed_new_tile = False  # Whether the current player has placed at least one new tile in this turn
        self.info = {}
        self.turn_number = 0
        return self._get_observation(), self.info  # Return observation and info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # Check if game is already over
        if self.done:
            return self._get_observation(), reward, terminated, truncated, self.info

        if action == self.rearrange_action_code:
            # Rearrangement action
            # For the purposes of this environment, rearrangement does nothing
            # In a real game, we would allow rearrangement of the equation space
            pass  # No action taken for rearrangement
        elif action == self.end_turn_action_code:
            # End turn action
            if not self.placed_new_tile:
                # Cannot end turn without placing at least one new tile
                reward = -10
                terminated = True
                self.done = True
                return self._get_observation(), reward, terminated, truncated, self.info
            else:
                # Switch to the next player
                self.current_player = 3 - self.current_player  # Switch between 1 and 2
                self.placed_new_tile = False
                self.turn_number += 1
                return self._get_observation(), reward, terminated, truncated, self.info
        else:
            # Place tile action
            tile_index = action // self.max_eq_length
            position = action % self.max_eq_length

            # Validate tile index
            if tile_index < 0 or tile_index >= self.num_total_tiles:
                reward = -10
                terminated = True
                self.done = True
                return self._get_observation(), reward, terminated, truncated, self.info

            # Validate position
            if position < 0 or position >= self.max_eq_length:
                reward = -10
                terminated = True
                self.done = True
                return self._get_observation(), reward, terminated, truncated, self.info

            # Check if position is empty
            if self.equation_space[position] != -1:
                reward = -10
                terminated = True
                self.done = True
                return self._get_observation(), reward, terminated, truncated, self.info

            # Check if tile is available (for number tiles)
            if tile_index <= 10:
                if self.tile_pool_counts[tile_index] <= 0:
                    reward = -10
                    terminated = True
                    self.done = True
                    return (
                        self._get_observation(),
                        reward,
                        terminated,
                        truncated,
                        self.info,
                    )
                else:
                    # Place the number tile
                    self.equation_space[position] = tile_index
                    self.tile_pool_counts[tile_index] -= 1
            else:
                # Place an operator tile (operator tiles are unlimited)
                self.equation_space[position] = tile_index

            # Mark that the player has placed at least one new tile in this turn
            self.placed_new_tile = True

        # After action, check if a valid equation is formed
        equation_str = self._get_equation_string()
        if self._is_valid_equation(equation_str):
            # Current player wins
            reward = 1
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, self.info

        # Check if all number tiles have been used and no valid equation formed
        if np.sum(self.tile_pool_counts) == 0 and not self.placed_new_tile:
            # Last player loses
            reward = -1
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, self.info

        # Action was valid but game is not over yet
        return self._get_observation(), reward, terminated, truncated, self.info

    def render(self):
        # Return a visual representation of the equation space as a string
        equation_str = self._get_equation_string()
        return equation_str

    def valid_moves(self):
        # Return a list of integers of valid moves as indices of the action_space

        valid_actions = []

        # If game is over, no valid moves
        if self.done:
            return valid_actions

        # Actions for placing tiles
        for tile_index in range(self.num_total_tiles):
            # For number tiles, check if tile is available
            if tile_index <= 10 and self.tile_pool_counts[tile_index] <= 0:
                continue  # Skip unavailable number tiles
            # For operator tiles, no need to check availability (unlimited)

            # Positions in the equation space
            for position in range(self.max_eq_length):
                if self.equation_space[position] == -1:
                    action_code = tile_index * self.max_eq_length + position
                    valid_actions.append(action_code)

        # Rearrangement action
        valid_actions.append(self.rearrange_action_code)

        # End turn action (only if the player has placed at least one new tile)
        if self.placed_new_tile:
            valid_actions.append(self.end_turn_action_code)
        else:
            # According to the rules, must place at least one new tile before ending turn
            pass  # Do not include end turn action

        return valid_actions

    def _get_observation(self):
        # Combine equation_space and tile_pool_counts into a single observation
        observation = np.concatenate((self.equation_space, self.tile_pool_counts))
        return observation

    def _get_equation_string(self):
        # Build the equation string from the equation space
        equation_str = ""
        for idx in self.equation_space:
            if idx == -1:
                continue  # Skip empty positions
            symbol = self.tile_indices.get(idx)
            if symbol is not None:
                equation_str += symbol
        return equation_str

    def _is_valid_equation(self, equation_str):
        # Check if the equation is valid according to the game rules

        # Equation must contain an '=' sign
        if "=" not in equation_str:
            return False

        # Split the equation into LHS and RHS
        sides = equation_str.split("=")
        if len(sides) != 2:
            return False  # Invalid equation format

        lhs_str, rhs_str = sides[0], sides[1]
        # Check if both sides are non-empty
        if not lhs_str or not rhs_str:
            return False

        # Check for leading zeros in numbers
        if self._has_leading_zeros(lhs_str) or self._has_leading_zeros(rhs_str):
            return False

        # Evaluate both sides
        try:
            lhs_value = self._evaluate_expression(lhs_str)
            rhs_value = self._evaluate_expression(rhs_str)
        except Exception:
            return False  # Invalid mathematical expression

        # Compare the evaluated values
        return np.isclose(lhs_value, rhs_value, atol=1e-8)

    def _has_leading_zeros(self, expr_str):
        # Check for leading zeros in numbers
        tokens = self._tokenize_expression(expr_str)
        for token in tokens:
            if token.isdigit() and len(token) > 1 and token.startswith("0"):
                return True
        return False

    def _evaluate_expression(self, expr_str):
        # Evaluate the mathematical expression following the standard order of operations
        # Since no parentheses are used, we can use eval() with care
        # Replace division operator to ensure division by zero is handled properly

        expr_str = expr_str.replace("/", "/")
        # Replace any instances of '--' with '+'
        expr_str = expr_str.replace("--", "+")
        # Disallow division by zero
        if "/0" in expr_str:
            raise ZeroDivisionError("Division by zero")

        # Evaluate the expression
        value = eval(expr_str)
        return value

    def _tokenize_expression(self, expr_str):
        # Tokenize the expression into numbers and operators
        tokens = []
        current_token = ""
        for char in expr_str:
            if char.isdigit():
                current_token += char
            else:
                if current_token != "":
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
        if current_token != "":
            tokens.append(current_token)
        return tokens
