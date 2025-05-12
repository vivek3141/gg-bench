import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space consists of 25 discrete actions (positions on the 5x5 grid)
        self.action_space = spaces.Discrete(25)

        # The observation is the state of the grid, represented as a 1D array of size 25
        # Each cell can be -1 (Player 2's block), 0 (empty), or 1 (Player 1's block)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(25,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the grid to empty
        self.grid = np.zeros(25, dtype=np.int8)
        # Player 1 starts first (represented as 1)
        self.current_player = 1
        # Set the game status to not done
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is over, return the current state with reward 0
            return self.grid.copy(), 0, True, False, {}

        # Validate the action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move penalized with -10 and terminates the game
            self.done = True
            reward = -10
            return self.grid.copy(), reward, True, False, {}

        # Place the block on the grid
        self.grid[action] = self.current_player

        # Switch to the opponent to check if they have valid moves
        self.current_player *= -1  # Switch player

        opponent_valid_actions = self.valid_moves()
        if not opponent_valid_actions:
            # Opponent cannot move; current player wins
            self.done = True
            reward = 1  # Winning reward
            self.current_player *= -1  # Switch back to current player before returning
            return self.grid.copy(), reward, True, False, {}

        # Game continues; switch back to the current player for the next turn
        self.current_player *= -1
        reward = 0  # Valid move with no immediate win
        return self.grid.copy(), reward, False, False, {}

    def render(self):
        # Create a visual representation of the grid
        board_str = ""
        symbols = {1: "X", -1: "O", 0: "."}
        for i in range(5):
            row = ""
            for j in range(5):
                index = i * 5 + j
                cell = self.grid[index]
                row += symbols[cell] + " "
            board_str += row.strip() + "\n"
        return board_str.strip()

    def valid_moves(self):
        # Returns a list of valid action indices for the current player
        valid_actions = []
        opponent = -self.current_player
        for index in range(25):
            if self.grid[index] != 0:
                continue  # Skip if the cell is not empty
            row = index // 5
            col = index % 5
            is_adjacent_to_opponent = False
            # Check all adjacent cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # Skip the current cell
                    adj_row = row + dr
                    adj_col = col + dc
                    if 0 <= adj_row < 5 and 0 <= adj_col < 5:
                        adj_index = adj_row * 5 + adj_col
                        if self.grid[adj_index] == opponent:
                            is_adjacent_to_opponent = True
                            break
                if is_adjacent_to_opponent:
                    break
            if not is_adjacent_to_opponent:
                # Add to valid actions if not adjacent to opponent's block
                valid_actions.append(index)
        return valid_actions
