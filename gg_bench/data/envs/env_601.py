import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.int8)

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(16, dtype=np.int8)
        self.current_player = 1  # Player 1 starts and is represented by 1
        self.done = False
        self.sudden_death = False  # Indicates if the game is in Sudden Death phase
        return self.board.copy(), {}  # Return a copy of the board and empty info

    def step(self, action):
        info = {}
        if self.done:
            return self.board.copy(), 0, True, False, info

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self.board.copy(), -10, True, False, info

        # Apply action
        if not self.sudden_death:
            # Place marker on empty cell
            self.board[action] = self.current_player
        else:
            # Swap opponent's marker with current player's marker
            self.board[action] = self.current_player

        # Check for win condition
        if self.check_winner(self.current_player):
            self.done = True
            return self.board.copy(), 1, True, False, info

        # Check for Sudden Death entry
        if not self.sudden_death and np.all(self.board != 0):
            self.sudden_death = True

        # Switch player
        self.current_player *= -1
        return self.board.copy(), 0, False, False, info

    def render(self):
        # Generate a visual representation of the board
        marker_map = {1: "X", -1: "O", 0: "."}
        board_visual = "    A   B   C   D\n"
        board_visual += "  +" + "---+" * 4 + "\n"
        for i in range(4):
            row = f"{i+1} |"
            for j in range(4):
                index = i * 4 + j
                row += f" {marker_map[self.board[index]]} |"
            board_visual += row + "\n"
            board_visual += "  +" + "---+" * 4 + "\n"
        return board_visual

    def valid_moves(self):
        # Return a list of valid action indices based on game phase
        if not self.sudden_death:
            # Valid moves are empty cells
            return [i for i in range(16) if self.board[i] == 0]
        else:
            # Valid moves are opponent's markers
            return [i for i in range(16) if self.board[i] == -self.current_player]

    def check_winner(self, player):
        # Check all possible 2x2 squares for a winning condition
        board_2d = self.board.reshape(4, 4)
        for i in range(3):
            for j in range(3):
                square = board_2d[i : i + 2, j : j + 2]
                if np.all(square == player):
                    return True
        return False
