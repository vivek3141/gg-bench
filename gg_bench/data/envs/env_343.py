import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 18 possible actions (0-17)
        # Actions 0-8: '+' operator with numbers 1-9
        # Actions 9-17: '-' operator with numbers 1-9
        self.action_space = spaces.Discrete(18)

        # Observation space:
        # - Elements 0-8: Numbers 1-9 status
        #   - 0 if unused
        #   - +1 if used with '+'
        #   - -1 if used with '-'
        # - Element 9: Current total value (from -45 to +45)
        self.observation_space = spaces.Box(
            low=-45, high=45, shape=(10,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize game state
        self.numbers_state = np.zeros(9, dtype=np.int32)  # Numbers 1-9 status
        self.total_value = 0  # Current total value
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game over flag
        self.expression = ["0"]  # Start expression with '0'
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done:
            # If game is already over, do nothing
            return self._get_obs(), 0, self.done, False, {}

        # Map action to operator and number
        operator, number = self._action_to_operator_number(action)

        # Check if number is unused
        if self.numbers_state[number - 1] != 0:
            # Invalid action: number already used
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_obs(), reward, self.done, False, {}

        # Valid move
        # Update numbers_state
        if operator == "+":
            self.numbers_state[number - 1] = 1
            operator_sign = 1
        else:
            self.numbers_state[number - 1] = -1
            operator_sign = -1

        # Update total value
        self.total_value += operator_sign * number

        # Update expression
        self.expression.append(f" {operator} {number}")

        # Check for victory condition
        if self.total_value == 0:
            # Current player wins
            self.done = True
            reward = 1  # Reward for winning
            return self._get_obs(), reward, self.done, False, {}

        # Check if all numbers are used
        if np.all(self.numbers_state != 0):
            # No more numbers left
            self.done = True
            reward = -1  # Penalty for losing
            return self._get_obs(), reward, self.done, False, {}

        # Continue game
        reward = -10  # Penalty for each valid move
        self.current_player = 2 if self.current_player == 1 else 1  # Switch player
        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        # Return the current expression and total value as a string
        expression_str = "".join(self.expression)
        render_str = (
            f"Current Expression: {expression_str}\nCurrent Total: {self.total_value}\n"
        )
        return render_str

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for action in range(18):
            operator, number = self._action_to_operator_number(action)
            if self.numbers_state[number - 1] == 0:
                valid_actions.append(action)
        return valid_actions

    def _action_to_operator_number(self, action):
        # Map action index to operator and number
        if 0 <= action <= 8:
            operator = "+"
            number = action + 1  # Numbers 1-9 for actions 0-8
        elif 9 <= action <= 17:
            operator = "-"
            number = action - 9 + 1  # Numbers 1-9 for actions 9-17
        else:
            raise ValueError(f"Invalid action index: {action}")
        return operator, number

    def _get_obs(self):
        # Observation includes numbers_state and total_value
        obs = np.zeros(10, dtype=np.int32)
        obs[0:9] = self.numbers_state  # Numbers 1-9 status
        obs[9] = self.total_value  # Current total value
        return obs
