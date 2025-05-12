import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 cells to choose from
        self.action_space = spaces.Discrete(9)

        # Observation space consists of two arrays:
        # - cell_numbers: numbers assigned to each cell (1-9)
        # - ownership: 0=unclaimed, 1=Player 1, 2=Player 2
        # We'll flatten this into a single array of shape (18,)
        self.observation_space = spaces.Box(
            low=np.array([1] * 9 + [0] * 9),
            high=np.array([9] * 9 + [2] * 9),
            shape=(18,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the board with numbers from 1 to 9 shuffled randomly
        self.cell_numbers = np.arange(1, 10)
        self.np_random.shuffle(self.cell_numbers)

        # Ownership array: 0=unclaimed, 1=Player 1, 2=Player 2
        self.ownership = np.zeros(9, dtype=np.int32)

        # Current player: 1 or 2
        self.current_player = 1

        self.done = False

        # Create the observation
        observation = np.concatenate((self.cell_numbers, self.ownership))

        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_obs(), 0, True, False, {}

        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Claim the cell
        self.ownership[action] = self.current_player

        # Check for captures
        self._check_captures(action)

        # Check if all cells are claimed
        if np.all(self.ownership != 0):
            self.done = True
            # Game over, determine the winner
            winner = self._determine_winner()

            if winner == self.current_player:
                reward = 1  # Current player wins
            else:
                reward = 0  # Current player loses or tie
            return self._get_obs(), reward, True, False, {}
        else:
            # Switch to the other player
            self.current_player = 1 if self.current_player == 2 else 2
            return self._get_obs(), 0, False, False, {}

    def render(self):
        # Create a visual representation of the board
        symbols = {0: " ", 1: "X", 2: "O"}
        grid = ""
        for i in range(3):
            grid += "+---+---+---+\n"
            for j in range(3):
                idx = i * 3 + j
                owner = self.ownership[idx]
                number = self.cell_numbers[idx]
                if owner == 0:
                    cell = f" {number} "
                else:
                    cell = f" {symbols[owner]} "
                grid += f"|{cell}"
            grid += "|\n"
        grid += "+---+---+---+\n"
        return grid

    def valid_moves(self):
        # Return a list of indices of unclaimed cells
        return [i for i in range(9) if self.ownership[i] == 0]

    def _get_obs(self):
        # Combine cell_numbers and ownership into a single observation
        observation = np.concatenate((self.cell_numbers, self.ownership))
        return observation

    def _check_captures(self, action):
        # Get the indices of adjacent cells
        adjacent_indices = self._get_adjacent_indices(action)

        # Number of the newly claimed cell
        current_number = self.cell_numbers[action]

        for idx in adjacent_indices:
            if self.ownership[idx] != 0 and self.ownership[idx] != self.current_player:
                opponent_number = self.cell_numbers[idx]
                if current_number > opponent_number:
                    # Capture the opponent's cell
                    self.ownership[idx] = self.current_player

    def _get_adjacent_indices(self, index):
        # Compute row and column from index
        row = index // 3
        col = index % 3

        adjacent_indices = []

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip the current cell
                r = row + dr
                c = col + dc
                if 0 <= r < 3 and 0 <= c < 3:
                    adj_idx = r * 3 + c
                    adjacent_indices.append(adj_idx)

        return adjacent_indices

    def _determine_winner(self):
        # Count cells claimed by each player
        player1_cells = np.sum(self.ownership == 1)
        player2_cells = np.sum(self.ownership == 2)

        if player1_cells > player2_cells:
            return 1  # Player 1 wins
        elif player2_cells > player1_cells:
            return 2  # Player 2 wins
        else:
            # Tie-breaker: sum of numbers in claimed cells
            player1_sum = np.sum(self.cell_numbers[self.ownership == 1])
            player2_sum = np.sum(self.cell_numbers[self.ownership == 2])

            if player1_sum > player2_sum:
                return 1  # Player 1 wins
            elif player2_sum > player1_sum:
                return 2  # Player 2 wins
            else:
                # Final tie-breaker: highest numbered cell claimed
                player1_max = np.max(self.cell_numbers[self.ownership == 1], initial=0)
                player2_max = np.max(self.cell_numbers[self.ownership == 2], initial=0)

                if player1_max > player2_max:
                    return 1
                elif player2_max > player1_max:
                    return 2
                else:
                    return 0  # Tie
