import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 possible cells to move to (cells 0 to 8)
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # For each cell:
        # - Cell number (1-9)
        # - P1 current position flag (0 or 1)
        # - P2 current position flag (0 or 1)
        # - P1 visited flag (0 or 1)
        # - P2 visited flag (0 or 1)
        # Plus cumulative totals for P1 and P2
        # Total observation size: 9 cells * 5 features = 45 + 2 cumulative totals = 47
        self.observation_space = spaces.Box(
            low=np.array([1] * 9 + [0] * 38, dtype=np.float32),
            high=np.array([9] * 9 + [1] * 36 + [15, 15], dtype=np.float32),
            shape=(47,),
            dtype=np.float32,
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly assign numbers 1-9 to the grid cells
        self.grid_numbers = np.random.permutation(np.arange(1, 10))
        self.cumulative_total = {1: 0, 2: 0}  # Cumulative totals for P1 and P2
        self.positions = {1: None, 2: None}  # Positions of P1 and P2 (cell indices 0-8)
        self.visited = {
            1: np.zeros(9, dtype=np.float32),
            2: np.zeros(9, dtype=np.float32),
        }
        self.current_player = 1  # P1 starts the game
        self.step_count = 0  # To handle starting positions
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}

        # Get current player's data
        player = self.current_player
        opponent = 2 if player == 1 else 1
        position = self.positions[player]
        cumulative = self.cumulative_total[player]
        visited = self.visited[player]
        opponent_position = self.positions[opponent]

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Starting position selection
        if position is None:
            # Place the player on the selected edge cell
            self.positions[player] = action
            self.visited[player][action] = 1
            number = self.grid_numbers[action]
            cumulative += number
            self.cumulative_total[player] = cumulative
            # Check for win condition
            if cumulative == 15:
                self.done = True
                return self._get_observation(), 1, True, False, {}
            elif cumulative > 15:
                # This situation should not happen as we check valid_moves
                self.done = True
                return self._get_observation(), -10, True, False, {}

        else:
            # Move to adjacent cell
            self.positions[player] = action
            self.visited[player][action] = 1
            number = self.grid_numbers[action]
            cumulative += number
            self.cumulative_total[player] = cumulative
            # Check for win condition
            if cumulative == 15:
                self.done = True
                return self._get_observation(), 1, True, False, {}
            elif cumulative > 15:
                # This situation should not happen as we check valid_moves
                self.done = True
                return self._get_observation(), -10, True, False, {}

        # Check if the opponent can make a move
        self.current_player = opponent
        opponent_valid_moves = self.valid_moves()
        if not opponent_valid_moves:
            # Opponent cannot make a move, current player wins
            self.done = True
            self.current_player = player  # Switch back to current player
            return self._get_observation(), 1, True, False, {}

        # Switch back to current player for the next turn
        self.current_player = opponent
        return self._get_observation(), 0, False, False, {}

    def render(self):
        grid_symbols = []
        for i in range(9):
            cell_str = f"{int(self.grid_numbers[i])}"
            if self.positions[1] == i:
                cell_str = f"P1({cell_str})"
            elif self.positions[2] == i:
                cell_str = f"P2({cell_str})"
            elif self.visited[1][i]:
                cell_str = f"V1({cell_str})"
            elif self.visited[2][i]:
                cell_str = f"V2({cell_str})"
            grid_symbols.append(cell_str)

        grid_str = ""
        grid_str += "+-------+-------+-------+\n"
        for i in range(3):
            grid_str += "|"
            for j in range(3):
                idx = i * 3 + j
                cell = grid_symbols[idx]
                grid_str += f" {cell:^7} |"
            grid_str += "\n+-------+-------+-------+\n"

        grid_str += f"P1 Cumulative Total: {self.cumulative_total[1]}\n"
        grid_str += f"P2 Cumulative Total: {self.cumulative_total[2]}\n"
        grid_str += f"Current Player: {'P' + str(self.current_player)}\n"
        return grid_str

    def valid_moves(self):
        player = self.current_player
        position = self.positions[player]
        opponent_position = self.positions[2 if player == 1 else 1]
        cumulative = self.cumulative_total[player]
        visited = self.visited[player]

        valid_actions = []

        edge_cells = [
            0,
            1,
            2,
            3,
            5,
            6,
            7,
            8,
        ]  # All edge cells (excluding center cell 4)

        if position is None:
            # Starting position selection, can choose any unoccupied edge cell
            for cell in edge_cells:
                if self.positions[1] == cell or self.positions[2] == cell:
                    continue  # Cell already occupied
                number = self.grid_numbers[cell]
                if cumulative + number <= 15:
                    valid_actions.append(cell)
            return valid_actions

        # Get possible moves (up, down, left, right)
        row, col = divmod(position, 3)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_position = new_row * 3 + new_col
                if (
                    self.positions[1] == new_position
                    or self.positions[2] == new_position
                ):
                    continue  # Cell already occupied
                if self.visited[player][new_position]:
                    continue  # Already visited
                number = self.grid_numbers[new_position]
                if cumulative + number > 15:
                    continue  # Exceeds cumulative total
                valid_actions.append(new_position)
        return valid_actions

    def _get_observation(self):
        obs = np.zeros(47, dtype=np.float32)
        obs[0:9] = self.grid_numbers  # Cell numbers
        # P1 current position flag
        if self.positions[1] is not None:
            obs[9 + self.positions[1]] = 1.0
        # P2 current position flag
        if self.positions[2] is not None:
            obs[18 + self.positions[2]] = 1.0
        # P1 visited flags
        obs[27:36] = self.visited[1]
        # P2 visited flags
        obs[36:45] = self.visited[2]
        # Cumulative totals
        obs[45] = self.cumulative_total[1]
        obs[46] = self.cumulative_total[2]
        return obs
