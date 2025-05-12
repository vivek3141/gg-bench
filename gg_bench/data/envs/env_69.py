import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(25)  # 5x5 grid cells as actions
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=np.int8)

        # Initialize the grid and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)

        # Players' starting positions
        self.player_positions = {
            1: (0, 0),  # Player 1 starts at (0, 0)
            2: (4, 4),  # Player 2 starts at (4, 4)
        }

        # Mark starting positions on the grid
        self.grid[0, 0] = 1  # Player 1's position
        self.grid[4, 4] = 2  # Player 2's position

        # Visited cells
        self.visited = np.zeros((5, 5), dtype=bool)
        self.visited[0, 0] = True
        self.visited[4, 4] = True

        self.current_player = 1
        self.done = False

        return np.copy(self.grid), {}  # Observation and info

    def step(self, action):
        if self.done:
            return np.copy(self.grid), 0, True, False, {}

        # Map action to target position
        target_pos = self.action_to_position(action)
        current_pos = self.player_positions[self.current_player]

        # Check if the action is valid
        if not self.is_valid_move(current_pos, target_pos):
            self.done = True
            return np.copy(self.grid), -10, True, False, {}

        # Move the current player
        self.move_player(self.current_player, current_pos, target_pos)

        # Check for victory conditions
        if self.has_won(self.current_player):
            self.done = True
            return np.copy(self.grid), 1, True, False, {}

        # Check if the opponent has any valid moves
        opponent = 3 - self.current_player
        if not self.has_valid_moves(opponent):
            self.done = True
            return np.copy(self.grid), 1, True, False, {}

        # Switch to the opponent
        self.current_player = opponent

        return np.copy(self.grid), 0, False, False, {}

    def render(self):
        grid_str = ""
        for i in range(5):
            row_str = ""
            for j in range(5):
                cell_value = self.grid[i, j]
                if cell_value == 1:
                    cell = "P1"
                elif cell_value == 2:
                    cell = "P2"
                elif cell_value == -1:
                    cell = "* "
                else:
                    cell = ". "
                row_str += f"{cell} "
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self, player=None):
        if player is None:
            player = self.current_player
        moves = []
        current_pos = self.player_positions[player]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            row = current_pos[0] + dr
            col = current_pos[1] + dc
            if 0 <= row < 5 and 0 <= col < 5:
                if self.grid[row, col] == 0:
                    moves.append(self.position_to_action((row, col)))
        return moves

    def action_to_position(self, action):
        row = action // 5
        col = action % 5
        return (row, col)

    def position_to_action(self, pos):
        return pos[0] * 5 + pos[1]

    def is_valid_move(self, current_pos, target_pos):
        row, col = target_pos
        if not (0 <= row < 5 and 0 <= col < 5):
            return False
        if self.grid[row, col] != 0:
            return False
        # Check if the target cell is adjacent
        current_row, current_col = current_pos
        dr = abs(row - current_row)
        dc = abs(col - current_col)
        if (dr == 1 and dc == 0) or (dr == 0 and dc == 1):
            return True
        else:
            return False

    def move_player(self, player, current_pos, target_pos):
        # Mark the current position as visited
        self.grid[current_pos] = -1
        self.visited[current_pos] = True

        # Move player to the target position
        self.grid[target_pos] = player
        self.player_positions[player] = target_pos

    def has_won(self, player):
        opponent_start = self.get_start_position(3 - player)
        return self.player_positions[player] == opponent_start

    def has_valid_moves(self, player):
        current_pos = self.player_positions[player]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            row = current_pos[0] + dr
            col = current_pos[1] + dc
            if 0 <= row < 5 and 0 <= col < 5:
                if self.grid[row, col] == 0:
                    return True
        return False

    def get_start_position(self, player):
        return (0, 0) if player == 1 else (4, 4)
