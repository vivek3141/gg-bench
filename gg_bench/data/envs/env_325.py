import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # The action space is discrete, numbers 0 to 9 correspond to selecting numbers 1 to 10
        self.action_space = spaces.Discrete(10)

        # The observation is a vector of length 10
        # Each index represents a number from 1 to 10
        # Values: 0 (available), 1 (selected by Player 1), -1 (selected by Player 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board and current player
        self.board = np.zeros(10, dtype=np.int8)
        self.current_player = 1  # Player 1 starts the game
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        if action < 0 or action >= 10 or self.board[action] != 0:
            # Invalid move
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Valid move
        self.board[action] = self.current_player

        # Check for win
        player_numbers = (
            np.where(self.board == self.current_player)[0] + 1
        )  # Numbers selected by current player
        if self.check_win(player_numbers):
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch to the other player
        self.current_player *= -1

        # Since there is no possibility of a draw, the game continues until someone wins
        return self.board.copy(), 0, False, False, {}

    def check_win(self, numbers):
        # Check all combinations of three numbers for an arithmetic sequence
        for combo in combinations(numbers, 3):
            combo = sorted(combo)
            if combo[1] - combo[0] == combo[2] - combo[1]:
                # Found an arithmetic sequence
                return True
        return False

    def render(self):
        # Return a string representation of the game state
        available_numbers = [str(i + 1) for i in range(10) if self.board[i] == 0]
        player1_numbers = [str(i + 1) for i in range(10) if self.board[i] == 1]
        player2_numbers = [str(i + 1) for i in range(10) if self.board[i] == -1]

        state_str = f"Available Numbers: {', '.join(available_numbers)}\n"
        state_str += f"Player 1's Numbers: {', '.join(player1_numbers)}\n"
        state_str += f"Player 2's Numbers: {', '.join(player2_numbers)}\n"
        return state_str

    def valid_moves(self):
        # Return a list of valid moves (indices of available numbers)
        return [i for i in range(10) if self.board[i] == 0]
