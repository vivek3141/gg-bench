import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros(16, dtype=np.int8)
        self.current_player = 1  # Player 1 uses '1', Player 2 uses '-1'
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        # Check if the current player has any valid moves
        valid_actions = self.valid_moves(self.current_player)
        if not valid_actions:
            # Current player has no valid moves and loses the game
            self.done = True
            return self.grid.copy(), -1, True, False, {}  # Lose condition

        if action not in valid_actions:
            self.done = True
            return self.grid.copy(), -10, True, False, {}  # Invalid move loses the game

        # Place the marker on the grid
        self.grid[action] = self.current_player

        # Check if the opponent has any valid moves
        opponent = -self.current_player
        opponent_valid_moves = self.valid_moves(opponent)
        if not opponent_valid_moves:
            # Opponent has no valid moves; current player wins
            self.done = True
            return self.grid.copy(), 1, True, False, {}  # Win condition

        # Switch to the opponent
        self.current_player = opponent
        return self.grid.copy(), 0, False, False, {}  # Continue the game

    def render(self):
        output = ""
        output += "-----------------\n"
        for row in range(4):
            output += "|"
            for col in range(4):
                val = self.grid[row * 4 + col]
                if val == 1:
                    output += " X |"
                elif val == -1:
                    output += " O |"
                else:
                    output += "   |"
            output += "\n-----------------\n"
        return output

    def valid_moves(self, player=None):
        if player is None:
            player = self.current_player
        valid_moves = []
        for cell in range(16):
            if self.grid[cell] != 0:
                continue  # Cell is not empty
            # Get adjacent cells
            adj_cells = self.get_adjacent_cells(cell)
            # Check if any adjacent cell contains player's marker
            adjacent_contains_player_marker = False
            for adj in adj_cells:
                if self.grid[adj] == player:
                    adjacent_contains_player_marker = True
                    break
            if not adjacent_contains_player_marker:
                valid_moves.append(cell)
        return valid_moves

    def get_adjacent_cells(self, cell):
        row = cell // 4
        col = cell % 4
        adjacent_cells = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip the cell itself
                r, c = row + dr, col + dc
                if 0 <= r < 4 and 0 <= c < 4:
                    adj_cell = r * 4 + c
                    adjacent_cells.append(adj_cell)
        return adjacent_cells
