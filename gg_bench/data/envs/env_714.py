import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        # Actions 0-9: Cross out single numbers 1-10
        # Actions 10-18: Cross out consecutive pairs 1-2, 2-3, ..., 9-10
        self.action_space = spaces.Discrete(19)

        # Observation space: 10 elements representing numbers 1 to 10
        # 0: Unmarked, 1: Crossed out
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the number line: all numbers are unmarked (0)
        self.number_line = np.zeros(10, dtype=np.int8)
        # Player 1 starts (can be 1 or -1)
        self.current_player = 1
        # Game is not over
        self.done = False
        # Return initial observation and empty info
        return self.number_line.copy(), {}

    def step(self, action):
        if self.done:
            # If game is over, return current state
            return self.number_line.copy(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move attempted
            self.done = True
            # Negative reward for invalid move
            return self.number_line.copy(), -10, True, False, {}

        # Map action to numbers to cross out
        if 0 <= action <= 9:
            # Single number action
            num = action  # Index from 0 to 9
            if self.number_line[num] == 0:
                self.number_line[num] = 1
            else:
                # Number already crossed out
                self.done = True
                return self.number_line.copy(), -10, True, False, {}
        elif 10 <= action <= 18:
            # Two consecutive numbers action
            num = action - 10  # Starting index from 0 to 8
            if self.number_line[num] == 0 and self.number_line[num + 1] == 0:
                self.number_line[num] = 1
                self.number_line[num + 1] = 1
            else:
                # One or both numbers already crossed out
                self.done = True
                return self.number_line.copy(), -10, True, False, {}
        else:
            # Invalid action index
            self.done = True
            return self.number_line.copy(), -10, True, False, {}

        # Check if the next player has any valid moves
        next_player_valid_moves = self.valid_moves()
        if len(next_player_valid_moves) == 0:
            # Current player wins
            self.done = True
            return self.number_line.copy(), 1, True, False, {}
        else:
            # Switch to next player
            self.current_player *= -1
            # Game continues
            return self.number_line.copy(), 0, False, False, {}

    def render(self):
        # Create a visual representation of the number line
        line_representation = ""
        for i in range(10):
            if self.number_line[i] == 0:
                line_representation += f"{i + 1} "
            else:
                line_representation += "X "
        return line_representation.strip()

    def valid_moves(self):
        # Generate list of valid action indices for current player
        moves = []
        # Single number actions
        for i in range(10):
            if self.number_line[i] == 0:
                moves.append(i)
        # Two consecutive numbers actions
        for i in range(9):
            if self.number_line[i] == 0 and self.number_line[i + 1] == 0:
                moves.append(10 + i)
        return moves
