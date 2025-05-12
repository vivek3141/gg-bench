import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to positions 0 to 6 (1 to 7 in game)
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.ones(7, dtype=np.int8)  # Initialize all elements to 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, return current state with zero reward
            return self.board, 0, True, False, {}

        if action < 0 or action > 6:
            # Invalid action handling
            self.done = True
            return self.board, -10, True, False, {}

        # Determine indices to flip based on the selected position
        if action == 0:
            # Selecting position 1 flips positions 0 and 1
            indices_to_flip = [0, 1]
        elif action == 6:
            # Selecting position 7 flips positions 5 and 6
            indices_to_flip = [5, 6]
        else:
            # Selecting positions 2 to 6 flips the position and its neighbors
            indices_to_flip = [action - 1, action, action + 1]

        # Flip the elements
        for idx in indices_to_flip:
            self.board[idx] = 1 - self.board[idx]  # Flip 0 to 1 and 1 to 0

        # Check for a win condition
        if np.all(self.board == 0):
            self.done = True
            reward = 1  # Current player wins
            return self.board, reward, True, False, {}
        else:
            reward = -10  # Valid move but game continues
            # Switch to the other player
            self.current_player = 1 if self.current_player == 2 else 2
            return self.board, reward, False, False, {}

    def render(self):
        # Build a string representation of the current state
        positions = " ".join(f"[{i+1}]" for i in range(7))
        values = " ".join(f" {self.board[i]} " for i in range(7))
        board_str = f"Positions: {positions}\nValues:    {values}\n"
        return board_str

    def valid_moves(self):
        # All positions are always valid moves
        return list(range(7))
