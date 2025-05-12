import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_length=6):
        super(CustomEnv, self).__init__()

        # The target length of the expression
        self.target_length = target_length

        # Action space: 0 for '(', 1 for ')'
        self.action_space = spaces.Discrete(2)

        # Observation space: array of length target_length
        # Each entry can be 0 (empty), 1 ('('), or 2 (')')
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.target_length,), dtype=np.int8
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.expression = np.zeros(self.target_length, dtype=np.int8)
        self.current_position = 0  # Next position to fill
        self.num_open = 0
        self.num_close = 0
        self.current_player = 0  # Player 0 starts
        self.done = False

        return self.expression.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.expression.copy(), 0, True, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return self.expression.copy(), reward, True, False, {}

        # Apply the action
        if action == 0:
            # Add '('
            self.expression[self.current_position] = 1
            self.num_open += 1
        elif action == 1:
            # Add ')'
            self.expression[self.current_position] = 2
            self.num_close += 1

        self.current_position += 1  # Move to next position

        # Check for win/loss conditions

        # If the expression is complete
        if self.current_position == self.target_length:
            # Check if the expression is balanced
            if self.num_open == self.num_close:
                # Current player wins
                self.done = True
                reward = 1  # Reward for winning
                return self.expression.copy(), reward, True, False, {}
            else:
                # Expression is unbalanced at the end
                self.done = True
                reward = -10  # Penalty for unbalanced expression
                return self.expression.copy(), reward, True, False, {}

        # If the opponent cannot make a valid move
        # Switch to next player temporarily to check valid moves
        self.current_player = 1 - self.current_player
        opponent_valid_actions = self.valid_moves()
        self.current_player = 1 - self.current_player  # Switch back

        if len(opponent_valid_actions) == 0:
            # Opponent cannot make a valid move
            # Current player wins
            self.done = True
            reward = 1  # Reward for winning
            return self.expression.copy(), reward, True, False, {}

        # Switch to next player for the next turn
        self.current_player = 1 - self.current_player

        # Continue the game
        reward = 0  # No reward yet
        return self.expression.copy(), reward, False, False, {}

    def render(self):
        # Convert the expression array into a string
        expr_str = ""
        for token in self.expression:
            if token == 0:
                expr_str += "_"
            elif token == 1:
                expr_str += "("
            elif token == 2:
                expr_str += ")"
        return f"Current Expression: {expr_str}"

    def valid_moves(self):
        # List of valid actions for the current player
        valid_actions = []
        # If we have reached target length, no moves are valid
        if self.current_position >= self.target_length:
            return valid_actions

        # Can always add '(' if we have not reached target length
        valid_actions.append(0)

        # Can add ')' if there is at least one unmatched '('
        if self.num_open > self.num_close:
            valid_actions.append(1)

        return valid_actions
