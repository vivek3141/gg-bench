import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 16 possible cells to claim
        self.action_space = spaces.Discrete(16)

        # Define observation space
        # Observation is a (16, 2) array where:
        # - First column: Ownership status (-1 for Player 2, 0 for unclaimed, 1 for Player 1)
        # - Second column: Cell number (1 to 16)
        self.low = np.array([[-1, 1]] * 16, dtype=np.int8)
        self.high = np.array([[1, 16]] * 16, dtype=np.int8)
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.int8)

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly assign numbers from 1 to 16 to the cells
        self.cell_numbers = np.random.permutation(np.arange(1, 17))
        # Initialize ownership: 0 for unclaimed
        self.ownership = np.zeros(16, dtype=np.int8)
        # Set current player: 1 for Player 1, -1 for Player 2
        self.current_player = 1
        self.done = False
        # Prepare observation
        self.observation = np.column_stack((self.ownership, self.cell_numbers))
        return self.observation, {}  # Return observation and info

    def step(self, action):
        # Check for valid action
        if action < 0 or action >= 16 or self.ownership[action] != 0 or self.done:
            # Invalid move
            self.done = True
            return (
                self.observation,
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Claim the selected cell
        self.ownership[action] = self.current_player
        claimed_number = self.cell_numbers[action]

        # Get row and column of the action
        row = action // 4
        col = action % 4

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            adj_row = row + dr
            adj_col = col + dc
            if 0 <= adj_row < 4 and 0 <= adj_col < 4:
                adj_index = adj_row * 4 + adj_col
                if self.ownership[adj_index] == -self.current_player:
                    adj_number = self.cell_numbers[adj_index]
                    if adj_number < claimed_number:
                        # Capture the opponent's cell
                        self.ownership[adj_index] = self.current_player

        # Update observation
        self.observation = np.column_stack((self.ownership, self.cell_numbers))

        # Check if the game is over
        if np.all(self.ownership != 0):
            self.done = True
            # Compute scores
            player1_cells = self.cell_numbers[self.ownership == 1]
            player2_cells = self.cell_numbers[self.ownership == -1]
            player1_score = np.sum(player1_cells)
            player2_score = np.sum(player2_cells)
            if player1_score > player2_score:
                winner = 1
            else:
                winner = -1
            # Assign reward
            if winner == self.current_player:
                reward = 1
            else:
                reward = -1
            return self.observation, reward, True, False, {}
        else:
            # Switch current player
            self.current_player *= -1
            return self.observation, 0, False, False, {}

    def render(self):
        # Create a visual representation of the board
        grid = ""
        for i in range(4):
            grid += "+----" * 4 + "+\n"
            for j in range(4):
                index = i * 4 + j
                owner = self.ownership[index]
                number = self.cell_numbers[index]
                if owner == 1:
                    cell_str = f"A{number:2d}"
                elif owner == -1:
                    cell_str = f"B{number:2d}"
                else:
                    cell_str = f" {number:2d}"
                grid += f"|{cell_str}"
            grid += "|\n"
        grid += "+----" * 4 + "+\n"
        return grid

    def valid_moves(self):
        # Return a list of unclaimed cell indices
        return [i for i in range(16) if self.ownership[i] == 0]
