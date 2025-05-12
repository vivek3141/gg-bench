import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 225 place actions + 300 swap actions = 525 total actions
        self.action_space = spaces.Discrete(525)
        # Observation space: 25 grid cells + 9 numbers in the pool = 34
        # Grid cells range from -9 to 9 (negative for Player 2)
        self.observation_space = spaces.Box(low=-9, high=9, shape=(34,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid
        self.grid = np.zeros((5, 5), dtype=np.int32)
        # Initialize the number pool: numbers 1-9, each with 3 copies
        self.number_pool = np.array([3] * 9, dtype=np.int32)
        # Track if swap action has been used by each player
        self.swap_used = {1: False, -1: False}
        # Start with Player 1
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        # Flatten the grid and concatenate with the number pool
        obs_grid = self.grid.flatten()
        obs = np.concatenate([obs_grid, self.number_pool])
        return obs

    def step(self, action):
        if self.done:
            return (
                self._get_obs(),
                -10,
                True,
                False,
                {},
            )  # Invalid move after game is over

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid move

        reward = 0

        if action < 225:
            # Place action
            number = (action // 25) + 1  # Numbers from 1 to 9
            cell_index = action % 25
            row, col = divmod(cell_index, 5)
            self.grid[row, col] = number * self.current_player
            self.number_pool[number - 1] -= 1
        else:
            # Swap action
            swap_index = action - 225
            cell_index1, cell_index2 = self._swap_index_to_cells(swap_index)
            row1, col1 = divmod(cell_index1, 5)
            row2, col2 = divmod(cell_index2, 5)
            # Swap positions
            self.grid[row1, col1], self.grid[row2, col2] = (
                self.grid[row2, col2],
                self.grid[row1, col1],
            )
            self.swap_used[self.current_player] = True

        # Check for win condition
        if self._check_win():
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}

        # Check for draw
        if not np.any(self.grid == 0) and not np.any(self.number_pool > 0):
            self.done = True
            winner = self._determine_winner()
            if winner == self.current_player:
                reward = 1
            else:
                reward = 0
            return self._get_obs(), reward, True, False, {}

        # Switch player
        self.current_player *= -1
        return self._get_obs(), reward, False, False, {}

    def render(self):
        grid_str = "   " + "  ".join(["1 ", "2 ", "3 ", "4 ", "5"]) + "\n"
        for i in range(5):
            row_str = chr(65 + i) + " "
            for j in range(5):
                value = self.grid[i, j]
                if value == 0:
                    cell_str = "[   ]"
                elif value > 0:
                    cell_str = f"[X{value}]"
                else:
                    cell_str = f"[O{-value}]"
                row_str += cell_str
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        valid_actions = []
        # Place actions
        for num in range(1, 10):  # Numbers from 1 to 9
            if self.number_pool[num - 1] > 0:
                for cell_index in range(25):  # Cells from 0 to 24
                    row, col = divmod(cell_index, 5)
                    if self.grid[row, col] == 0:
                        action = (num - 1) * 25 + cell_index
                        valid_actions.append(action)
        # Swap actions
        if not self.swap_used[self.current_player]:
            for swap_index in range(300):
                cell_index1, cell_index2 = self._swap_index_to_cells(swap_index)
                action = 225 + swap_index
                valid_actions.append(action)
        return valid_actions

    def _swap_index_to_cells(self, swap_index):
        # Map swap_index to (cell_index1, cell_index2)
        count = 0
        for i in range(25):
            for j in range(i + 1, 25):
                if count == swap_index:
                    return i, j
                count += 1
        raise ValueError(f"Invalid swap_index {swap_index}")

    def _cells_to_swap_index(self, cell_index1, cell_index2):
        # Map (cell_index1, cell_index2) to swap_index
        if cell_index1 > cell_index2:
            cell_index1, cell_index2 = cell_index2, cell_index1
        count = 0
        for i in range(25):
            for j in range(i + 1, 25):
                if i == cell_index1 and j == cell_index2:
                    return count
                count += 1
        raise ValueError(f"Invalid cell indices {cell_index1}, {cell_index2}")

    def _check_win(self):
        # Check if the current player has formed a sequence
        player_grid = np.where(
            self.grid * self.current_player > 0, self.grid * self.current_player, 0
        )

        # Check rows
        for row in player_grid:
            if self._check_sequence(row):
                return True

        # Check columns
        for col in player_grid.T:
            if self._check_sequence(col):
                return True

        # Check diagonals
        diag1 = player_grid.diagonal()
        diag2 = np.fliplr(player_grid).diagonal()
        if self._check_sequence(diag1) or self._check_sequence(diag2):
            return True

        return False

    def _check_sequence(self, line):
        # Check for ascending or descending sequences of three numbers
        n = len(line)
        for i in range(n - 2):
            seq = line[i : i + 3]
            if np.all(seq > 0):
                if seq[0] < seq[1] < seq[2] or seq[0] > seq[1] > seq[2]:
                    return True
        return False

    def _determine_winner(self):
        # Determine the winner based on potential sequences of two
        potentials = {1: 0, -1: 0}
        for player in [1, -1]:
            player_grid = np.where(self.grid * player > 0, self.grid * player, 0)
            # Rows
            for row in player_grid:
                potentials[player] += self._count_potentials(row)
            # Columns
            for col in player_grid.T:
                potentials[player] += self._count_potentials(col)
            # Diagonals
            diag1 = player_grid.diagonal()
            diag2 = np.fliplr(player_grid).diagonal()
            potentials[player] += self._count_potentials(diag1)
            potentials[player] += self._count_potentials(diag2)

        if potentials[1] > potentials[-1]:
            return 1
        elif potentials[-1] > potentials[1]:
            return -1
        else:
            # Tie-breaker: player with more numbers on the grid
            counts = {1: np.sum(self.grid > 0), -1: np.sum(self.grid < 0)}
            if counts[1] > counts[-1]:
                return 1
            elif counts[-1] > counts[1]:
                return -1
            else:
                return 0  # It's a tie

    def _count_potentials(self, line):
        # Count potential sequences of two numbers
        count = 0
        n = len(line)
        for i in range(n - 1):
            seq = line[i : i + 2]
            if np.all(seq > 0):
                if seq[0] < seq[1] or seq[0] > seq[1]:
                    count += 1
        return count
