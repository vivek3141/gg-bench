import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 81 actions for the placement phase and 36 actions for the sudden death phase
        # Total actions = 81 (placement) + 36 (swap) = 117
        self.action_space = spaces.Discrete(117)
        self.observation_space = spaces.Box(low=-9, high=9, shape=(9,), dtype=np.int8)

        # Game state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid
        self.grid = np.zeros(9, dtype=np.int8)  # 0 indicates empty cell
        self.unclaimed_numbers = list(range(1, 10))  # Numbers 1-9 unclaimed
        self.current_player = 1  # Player 1 starts (1 for Player 1, -1 for Player 2)
        self.done = False
        self.phase = "placement"  # Game phase: 'placement' or 'sudden_death'
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        info = {}
        reward = 0

        # Check if game is already over
        if self.done:
            return self._get_obs(), reward, self.done, False, info

        valid_actions = self.valid_moves()

        # Check if action is valid
        if action not in valid_actions:
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_obs(), reward, self.done, False, info

        if self.phase == "placement":
            # Map action to (number, cell)
            number_index = action // 9  # 0-8
            cell_index = action % 9  # 0-8
            number = number_index + 1  # Numbers 1-9

            # Place the number on the grid
            self.grid[cell_index] = self.current_player * number
            self.unclaimed_numbers.remove(number)

            # Check for winning condition
            if self._check_winner():
                self.done = True
                reward = 1  # Reward for winning
                return self._get_obs(), reward, self.done, False, info

            # Check if all numbers have been placed
            if len(self.unclaimed_numbers) == 0:
                # Transition to sudden death phase
                self.phase = "sudden_death"
            else:
                # Switch to the other player
                self.current_player *= -1
        elif self.phase == "sudden_death":
            # Map action to swap indices
            swap_index = action - 81  # 0-35
            swap_pairs = self._get_swap_pairs()
            cell1, cell2 = swap_pairs[swap_index]

            # Swap the numbers in the cells
            self.grid[cell1], self.grid[cell2] = self.grid[cell2], self.grid[cell1]

            # Check for winning condition
            if self._check_winner():
                self.done = True
                reward = 1  # Reward for winning
                return self._get_obs(), reward, self.done, False, info

            # Switch to the other player
            self.current_player *= -1

        return self._get_obs(), reward, self.done, False, info

    def render(self):
        board_str = "-------------\n"
        for i in range(3):
            board_str += "|"
            for j in range(3):
                idx = i * 3 + j
                cell_value = self.grid[idx]
                if cell_value == 0:
                    board_str += "   |"
                else:
                    num = abs(cell_value)
                    player = "P1" if cell_value > 0 else "P2"
                    board_str += f"{num:2}{player}|"
            board_str += "\n-------------\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        if self.phase == "placement":
            for number in self.unclaimed_numbers:
                number_index = number - 1  # 0-8
                for cell_index in range(9):
                    if self.grid[cell_index] == 0:
                        action = number_index * 9 + cell_index
                        valid_actions.append(action)
        elif self.phase == "sudden_death":
            swap_pairs = self._get_swap_pairs()
            for idx, (cell1, cell2) in enumerate(swap_pairs):
                action = 81 + idx
                valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        # Observation is the grid
        return self.grid.copy()

    def _get_swap_pairs(self):
        # Generate all unique pairs of cells to swap (cell1 < cell2)
        swap_pairs = []
        for i in range(9):
            for j in range(i + 1, 9):
                swap_pairs.append((i, j))
        return swap_pairs

    def _check_winner(self):
        winning_lines = [
            [0, 1, 2],  # Rows
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],  # Columns
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],  # Diagonals
            [2, 4, 6],
        ]

        for line in winning_lines:
            line_numbers = []
            line_owners = []
            for idx in line:
                cell_value = self.grid[idx]
                if cell_value != 0:
                    line_numbers.append(abs(cell_value))
                    line_owners.append(np.sign(cell_value))
                else:
                    break  # Incomplete line

            if len(line_numbers) == 3 and sum(line_numbers) == 15:
                # Check if current player has majority in this line
                player_count = line_owners.count(self.current_player)
                if player_count >= 2:
                    return True  # Current player wins
        return False
