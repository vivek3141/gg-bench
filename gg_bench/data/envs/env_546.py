import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Grid size
        self.grid_size = 4  # Default size is 4x4

        # Define action and observation space
        # There are grid_size * grid_size possible actions (one for each cell)
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)

        # Observation space is a grid_size x grid_size grid
        # Each cell can have a value between -10 and 10
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(self.grid_size, self.grid_size), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the hidden grid with random values between 1 and 10
        self.hidden_grid = np.random.randint(
            1, 11, size=(self.grid_size, self.grid_size)
        )

        # Initialize claimed grid: 0 = unclaimed, 1 = player 1, -1 = player 2
        self.claimed_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Initialize scores
        self.player1_score = 0
        self.player2_score = 0

        # Current player (1 for player 1, -1 for player 2)
        self.current_player = 1

        # Game over flag
        self.done = False

        # Return initial observation and info
        return self.get_observation(), {}

    def get_observation(self):
        # Returns the observation of the current state from the current player's perspective
        observation = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.claimed_grid[i][j] == self.current_player:
                    # Current player's claimed cell: positive value
                    observation[i][j] = self.hidden_grid[i][j]
                elif self.claimed_grid[i][j] == -self.current_player:
                    # Opponent's claimed cell: negative value
                    observation[i][j] = -self.hidden_grid[i][j]
                else:
                    # Unclaimed cell: 0 (value is hidden)
                    observation[i][j] = 0
        return observation

    def step(self, action):
        if self.done:
            return self.get_observation(), -10, True, False, {}  # Game is over

        # Convert action to grid coordinates
        row = action // self.grid_size
        col = action % self.grid_size

        # Check if the action is valid
        if self.claimed_grid[row][col] != 0:
            self.done = True
            return self.get_observation(), -10, True, False, {}  # Invalid move

        # Claim the cell
        self.claimed_grid[row][col] = self.current_player

        # Get the value of the cell
        cell_value = self.hidden_grid[row][col]

        # Update the current player's score
        if self.current_player == 1:
            self.player1_score += cell_value
        else:
            self.player2_score += cell_value

        # Adjust adjacent unclaimed cells
        adjacents = []
        if row > 0:
            adjacents.append((row - 1, col))  # Up
        if row < self.grid_size - 1:
            adjacents.append((row + 1, col))  # Down
        if col > 0:
            adjacents.append((row, col - 1))  # Left
        if col < self.grid_size - 1:
            adjacents.append((row, col + 1))  # Right

        for adj_row, adj_col in adjacents:
            if self.claimed_grid[adj_row][adj_col] == 0:
                # Reduce hidden value by 1, not going below 0
                self.hidden_grid[adj_row][adj_col] = max(
                    0, self.hidden_grid[adj_row][adj_col] - 1
                )

        # Check if the game is over (all cells claimed)
        if np.all(self.claimed_grid != 0):
            self.done = True
            # Determine the winner
            if self.player1_score > self.player2_score:
                winner = 1
            elif self.player2_score > self.player1_score:
                winner = -1
            else:
                winner = 0  # Draw

            if winner == self.current_player:
                reward = 1  # Current player wins
            else:
                reward = 0  # Loss or draw
            return self.get_observation(), reward, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        # Continue the game
        return self.get_observation(), 0, False, False, {}

    def render(self):
        # Return a visual representation of the game state as a string
        grid_str = ""
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                if self.claimed_grid[i][j] == 0:
                    row_str += " . "
                elif self.claimed_grid[i][j] == self.current_player:
                    # Current player's claimed cell: show value
                    value = self.hidden_grid[i][j]
                    row_str += f" {value:2d} "
                else:
                    # Opponent's claimed cell: show value in parentheses
                    value = self.hidden_grid[i][j]
                    row_str += f"({value:2d})"
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        # Return a list of valid action indices for unclaimed cells
        return [
            i * self.grid_size + j
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if self.claimed_grid[i][j] == 0
        ]
