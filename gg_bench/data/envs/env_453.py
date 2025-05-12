import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 0-17 are assignments, 18 is 'pass'
        self.action_space = spaces.Discrete(19)
        # Observation space: number pool (9), player 1 base/exponent (2), player 2 base/exponent (2), current player (1)
        self.observation_space = spaces.Box(low=0, high=9, shape=(14,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the number pool (1-9 available)
        self.number_pool = {
            i: 1 for i in range(1, 10)
        }  # 1 means available, 0 means taken
        # Initialize players' expressions
        self.players = {1: {"base": 0, "exponent": 0}, 2: {"base": 0, "exponent": 0}}
        # Set current player
        self.current_player = 1
        # Game not done
        self.done = False
        # Return observation and info
        return self._get_obs(), {}

    def step(self, action):
        # Check if game is already done
        if self.done:
            return self._get_obs(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        if action == 18:  # 'pass' action
            if not self._expression_complete(self.current_player):
                # Cannot pass if expression is incomplete
                self.done = True
                return self._get_obs(), -10, True, False, {}
            else:
                # Pass turn to the next player
                self.current_player = 3 - self.current_player
                # Check if both expressions are complete
                if self._expressions_complete():
                    # Calculate results
                    reward = self._calculate_results()
                    self.done = True
                    return self._get_obs(), reward, True, False, {}
                else:
                    # Continue game
                    return self._get_obs(), 0, False, False, {}
        else:
            # Map action to number and position
            number = (action // 2) + 1
            position = "base" if action % 2 == 0 else "exponent"

            # Check if number is in pool and position is unassigned
            if (
                self.number_pool.get(number, 0) == 0
                or self.players[self.current_player][position] != 0
            ):
                self.done = True
                return self._get_obs(), -10, True, False, {}
            else:
                # Assign number to position
                self.players[self.current_player][position] = number
                # Remove number from pool
                self.number_pool[number] = 0

            # Check if both expressions are complete
            if self._expressions_complete():
                # Calculate results
                reward = self._calculate_results()
                self.done = True
                return self._get_obs(), reward, True, False, {}
            else:
                # If current player's expression is complete, pass turn automatically
                if self._expression_complete(self.current_player):
                    self.current_player = 3 - self.current_player
                return self._get_obs(), 0, False, False, {}

    def render(self):
        pool_numbers = [str(num) for num in range(1, 10) if self.number_pool[num] == 1]
        pool_str = "Number Pool: [" + ", ".join(pool_numbers) + "]\n"

        p1_base = self.players[1]["base"] if self.players[1]["base"] != 0 else "__"
        p1_exp = (
            self.players[1]["exponent"] if self.players[1]["exponent"] != 0 else "__"
        )
        p1_expr = f"Player 1's Expression: {p1_base} ^ {p1_exp}\n"

        p2_base = self.players[2]["base"] if self.players[2]["base"] != 0 else "__"
        p2_exp = (
            self.players[2]["exponent"] if self.players[2]["exponent"] != 0 else "__"
        )
        p2_expr = f"Player 2's Expression: {p2_base} ^ {p2_exp}\n"

        current_player_str = f"Current Player: Player {self.current_player}\n"

        state_str = pool_str + p1_expr + p2_expr + current_player_str
        return state_str

    def valid_moves(self):
        valid_actions = []
        if self._expression_complete(self.current_player):
            # Only 'pass' action is valid
            valid_actions.append(18)
        else:
            # For each number in pool
            for num in range(1, 10):
                if self.number_pool[num] == 1:
                    # For each unassigned position
                    for pos_idx, pos in enumerate(["base", "exponent"]):
                        if self.players[self.current_player][pos] == 0:
                            action = (num - 1) * 2 + pos_idx
                            valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        # Number Pool availability
        pool_availability = [self.number_pool[i] for i in range(1, 10)]
        # Players' expressions
        p1_base = self.players[1]["base"]
        p1_exp = self.players[1]["exponent"]
        p2_base = self.players[2]["base"]
        p2_exp = self.players[2]["exponent"]
        # Current player
        current_player = self.current_player
        obs = np.array(
            pool_availability + [p1_base, p1_exp, p2_base, p2_exp, current_player],
            dtype=np.int32,
        )
        return obs

    def _expression_complete(self, player):
        expr = self.players[player]
        return expr["base"] != 0 and expr["exponent"] != 0

    def _expressions_complete(self):
        return self._expression_complete(1) and self._expression_complete(2)

    def _calculate_results(self):
        # Calculate expressions
        expr1 = self.players[1]
        expr2 = self.players[2]
        result1 = expr1["base"] ** expr1["exponent"]
        result2 = expr2["base"] ** expr2["exponent"]

        # Determine winner
        if result1 > result2:
            winner = 1
        elif result2 > result1:
            winner = 2
        else:
            # Tie, player who completed second wins
            # Since current_player just completed their expression
            winner = self.current_player

        if winner == self.current_player:
            return 1  # Current player wins
        else:
            return -1  # Current player loses
