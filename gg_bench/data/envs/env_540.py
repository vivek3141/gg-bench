import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 16 cells in the matrix (4x4 grid), so action_space is Discrete(16)
        self.action_space = spaces.Discrete(16)
        # Observation space is a 4x4 grid of integers from 0 to 9
        # Available cells have values from 1 to 9, unavailable cells are marked with 0
        self.observation_space = spaces.Box(low=0, high=9, shape=(4, 4), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generate the 4x4 matrix with random integers from 1 to 9
        self.matrix = np.random.randint(1, 10, size=(4, 4))
        # Rows and columns that are currently unavailable (start with none)
        self.unavailable_rows = []
        self.unavailable_cols = []
        # Players' total sums
        self.player_sums = {1: 0, 2: 0}
        # Current player (1 or 2), Player 1 starts
        self.current_player = 1
        # Game over flag
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Convert action to row and col
        row = action // 4
        col = action % 4

        # Check if the selected cell is valid
        if (
            (row in self.unavailable_rows)
            or (col in self.unavailable_cols)
            or (self.matrix[row][col] == 0)
        ):
            # Invalid move
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}
        else:
            # Valid move
            # Add the number to the player's sum
            number = self.matrix[row][col]
            self.player_sums[self.current_player] += number

            # Remove the row and column from play
            self.unavailable_rows.append(row)
            self.unavailable_cols.append(col)

            # Set the values in the removed row and column to 0
            self.matrix[row, :] = 0
            self.matrix[:, col] = 0

            # Check if the game is over (no valid moves left)
            if not self.valid_moves():
                self.done = True
                p1_sum = self.player_sums[1]
                p2_sum = self.player_sums[2]

                # Determine the winner
                if p1_sum > p2_sum:
                    winner = 1
                elif p2_sum > p1_sum:
                    winner = 2
                else:
                    # Tie-breaker: Player 2 wins in case of a tie
                    winner = 2

                if winner == self.current_player:
                    reward = 1  # Current player wins
                else:
                    reward = 0  # Current player loses

                return self._get_observation(), reward, True, False, {}
            else:
                # Switch to the other player
                self.current_player = 1 if self.current_player == 2 else 2
                reward = 0
                return self._get_observation(), reward, False, False, {}

    def render(self):
        # Generate a string representation of the matrix
        grid = ""
        grid += "     C1  C2  C3  C4\n"
        for i in range(4):
            grid += f"R{i+1}: "
            for j in range(4):
                if (
                    (i in self.unavailable_rows)
                    or (j in self.unavailable_cols)
                    or self.matrix[i][j] == 0
                ):
                    grid += " X  "
                else:
                    grid += f" {self.matrix[i][j]}  "
            grid += "\n"
        grid += f"\nPlayer 1 Total Sum: {self.player_sums[1]}"
        grid += f"\nPlayer 2 Total Sum: {self.player_sums[2]}"
        grid += f"\nCurrent Player: Player {self.current_player}"
        return grid

    def valid_moves(self):
        # Return a list of actions (indices) that are valid
        valid_moves = []
        for action in range(16):
            row = action // 4
            col = action % 4
            if (
                (row not in self.unavailable_rows)
                and (col not in self.unavailable_cols)
                and (self.matrix[row][col] != 0)
            ):
                valid_moves.append(action)
        return valid_moves

    def _get_observation(self):
        # Return the current state of the matrix
        # For unavailable cells, we leave them as 0
        return self.matrix.copy()
