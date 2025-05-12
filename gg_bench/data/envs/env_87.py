import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define constants
        self.MAX_LEN = 20  # Maximum length of the bracket sequence

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: '(', 1: ')'
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.MAX_LEN,), dtype=np.int8
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = np.zeros(self.MAX_LEN, dtype=np.int8)
        self.current_index = 0  # Next position to insert bracket
        self.count_open = 0  # Number of '(' added
        self.count_close = 0  # Number of ')' added
        self.balance = 0  # count_open - count_close
        self.current_player = 1  # 1 or -1 to represent player
        self.done = False
        return self.sequence.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.sequence.copy(), 0.0, self.done, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self.sequence.copy(), -10.0, True, False, {}

        # Apply the action
        self.sequence[self.current_index] = action + 1  # 1 for '(', 2 for ')'
        self.current_index += 1

        if action == 0:
            self.count_open += 1
        else:
            self.count_close += 1

        # Update balance
        self.balance = self.count_open - self.count_close

        # Check for invalid sequence (more closing brackets than opening)
        if self.balance < 0:
            self.done = True
            return self.sequence.copy(), -10.0, True, False, {}

        # Check for victory (balanced sequence)
        if self.balance == 0:
            self.done = True
            return self.sequence.copy(), 1.0, True, False, {}

        # Check if the next player has no valid moves
        if not self.valid_moves():
            self.done = True
            return self.sequence.copy(), 1.0, True, False, {}

        # Switch player
        self.current_player *= -1

        return self.sequence.copy(), 0.0, False, False, {}

    def render(self):
        bracket_str = "".join(
            ["(" if x == 1 else ")" if x == 2 else "" for x in self.sequence]
        )
        return f"Current Sequence: {bracket_str}"

    def valid_moves(self):
        if self.done:
            return []

        moves = []
        # Can always add '(' if sequence is not full
        if self.current_index < self.MAX_LEN:
            moves.append(0)  # '('

        # Can add ')' if there are unmatched '('
        if self.count_open > self.count_close and self.current_index < self.MAX_LEN:
            moves.append(1)  # ')'

        return moves
