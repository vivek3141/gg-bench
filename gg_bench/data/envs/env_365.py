import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 possible actions, corresponding to cell positions 0-8
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # - 9 cells (0 for unrevealed, 1-9 for revealed numbers)
        # - 8 sum clues (rows, columns, diagonals)
        # Total of 17 elements
        self.observation_space = spaces.Box(
            low=0, high=25, shape=(17,), dtype=np.float32
        )

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly assign numbers 1-9 to the grid positions
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.numbers = self.np_random.permutation(np.arange(1, 10))
        self.grid_numbers = self.numbers.reshape((3, 3))

        # Compute sum clues
        self.row_sums = np.sum(self.grid_numbers, axis=1)
        self.col_sums = np.sum(self.grid_numbers, axis=0)
        self.diag1_sum = np.sum(np.diagonal(self.grid_numbers))
        self.diag2_sum = np.sum(np.diagonal(np.fliplr(self.grid_numbers)))

        # Initialize the grid to zeros (unrevealed cells)
        self.grid = np.zeros((3, 3), dtype=np.int32)

        # Initialize scores
        self.scores = {1: 0, -1: 0}

        # Player 1 starts (represented by 1)
        self.current_player = 1

        # Game not done
        self.done = False

        # Build initial observation
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        # Check if game is already done
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Convert action to grid coordinates
        if action < 0 or action >= 9:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {"invalid_move": True}
        row = action // 3
        col = action % 3

        # Check if the cell is already revealed
        if self.grid[row, col] != 0:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {"invalid_move": True}

        # Reveal the number in the selected cell
        number = self.grid_numbers[row, col]
        self.grid[row, col] = number

        # Update the current player's score
        self.scores[self.current_player] += number

        # Check if the game is over (all cells revealed)
        if np.all(self.grid != 0):
            self.done = True
            # Determine winner
            player_score = self.scores[self.current_player]
            opponent_score = self.scores[-self.current_player]
            if player_score > opponent_score:
                reward = 1  # Current player wins
            elif player_score < opponent_score:
                reward = -1  # Current player loses
            else:
                # In case of tie, the last player to move wins
                reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            # Switch current player
            self.current_player *= -1
            return self._get_observation(), reward, False, False, {}

    def render(self):
        # Build grid display
        grid_display = ""
        grid_positions = np.arange(1, 10).reshape((3, 3))
        for i in range(3):
            grid_display += "+-----+-----+-----+\n"
            for j in range(3):
                if self.grid[i, j] != 0:
                    cell_value = f"  {self.grid[i, j]}  "
                else:
                    cell_value = f" [{grid_positions[i,j]}] "
                grid_display += f"|{cell_value}"
            grid_display += "|\n"
        grid_display += "+-----+-----+-----+\n"

        # Display sum clues
        clues_display = (
            "Row sums:    "
            + ", ".join([f"Row {i+1}: {self.row_sums[i]}" for i in range(3)])
            + "\n"
        )
        clues_display += (
            "Column sums: "
            + ", ".join([f"Col {i+1}: {self.col_sums[i]}" for i in range(3)])
            + "\n"
        )
        clues_display += (
            f"Diagonal sums: Diag1: {self.diag1_sum}, Diag2: {self.diag2_sum}\n"
        )

        # Display scores
        scores_display = (
            f"Scores:\nPlayer 1: {self.scores[1]}\nPlayer 2: {self.scores[-1]}\n"
        )

        # Display current player
        player_display = f"Current player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"

        # Combine all displays
        full_display = grid_display + clues_display + scores_display + player_display
        return full_display

    def valid_moves(self):
        # Return list of indices (0-8) of unrevealed cells
        valid_moves = []
        for i in range(3):
            for j in range(3):
                if self.grid[i, j] == 0:
                    index = i * 3 + j
                    valid_moves.append(index)
        return valid_moves

    def _get_observation(self):
        # Flatten the grid
        grid_flat = self.grid.flatten()

        # Build the observation array
        observation = np.zeros(17, dtype=np.float32)
        observation[:9] = grid_flat
        observation[9:12] = self.row_sums
        observation[12:15] = self.col_sums
        observation[15] = self.diag1_sum
        observation[16] = self.diag2_sum

        return observation
