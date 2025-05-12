import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 13 possible actions:
        # Actions 0-6: Turn OFF one light (light at position index+1)
        # Actions 7-12: Turn OFF two adjacent lights:
        #   7: Turn OFF lights 1 and 2
        #   8: Turn OFF lights 2 and 3
        #   9: Turn OFF lights 3 and 4
        #   10: Turn OFF lights 4 and 5
        #   11: Turn OFF lights 5 and 6
        #   12: Turn OFF lights 6 and 7

        self.action_space = spaces.Discrete(13)
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.ones(7, dtype=np.int8)  # All lights are ON (1)
        self.current_player = 1  # Player 1 starts; can be 1 or -1
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        # Check if game is already over
        if self.done:
            return self.board.copy(), -10, True, False, {}

        # Check for valid moves
        valid_moves = self.valid_moves()
        if len(valid_moves) == 0:
            # No valid moves, current player loses
            self.done = True
            return self.board.copy(), -10, True, False, {}

        if action not in valid_moves:
            # Invalid action
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Apply the action
        if 0 <= action <= 6:
            # Single light action
            light_index = action
            if self.board[light_index] == 1:
                self.board[light_index] = 0
            else:
                # This should not happen due to valid_moves check
                self.done = True
                return self.board.copy(), -10, True, False, {}
        elif 7 <= action <= 12:
            # Double adjacent lights action
            pair_index = action - 7
            light1_index = pair_index
            light2_index = pair_index + 1
            if self.board[light1_index] == 1 and self.board[light2_index] == 1:
                self.board[light1_index] = 0
                self.board[light2_index] = 0
            else:
                # This should not happen due to valid_moves check
                self.done = True
                return self.board.copy(), -10, True, False, {}
        else:
            # Invalid action index
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Check if all lights are OFF after the action
        if np.all(self.board == 0):
            # Current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        return self.board.copy(), 0, False, False, {}

    def render(self):
        states = ["O" if x == 1 else "X" for x in self.board]
        board_str = "States: " + " ".join(states)
        return board_str

    def valid_moves(self):
        valid_actions = []
        # Single light OFF actions (actions 0-6)
        for i in range(7):
            if self.board[i] == 1:
                valid_actions.append(i)
        # Double adjacent lights OFF actions (actions 7-12)
        for i in range(6):
            if self.board[i] == 1 and self.board[i + 1] == 1:
                valid_actions.append(7 + i)
        return valid_actions
