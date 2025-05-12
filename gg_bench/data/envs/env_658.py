import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: 4 movement actions + 25 barrier placement actions = 29 total actions
        self.action_space = spaces.Discrete(29)

        # Observation space: 5x5 grid with values from 0 to 3
        # 0: empty cell, 1: Player 1 runner, 2: Player 2 runner, 3: barrier
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(5, 5), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.float32)
        self.grid[0, 0] = 1  # Player 1 starts at A1
        self.grid[4, 4] = 2  # Player 2 starts at E5

        # Each player starts with 3 barriers
        self.barriers = {1: 3, 2: 3}

        self.current_player = 1
        self.done = False

        return self.grid, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid, 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self.grid, -10, True, False, {}  # Invalid move

        if action < 4:
            # Movement action
            success = self.move_runner(self.current_player, action)
        else:
            # Barrier placement action
            cell_index = action - 4
            row = cell_index // 5
            col = cell_index % 5
            success = self.place_barrier(self.current_player, row, col)

        if not success:
            self.done = True
            return self.grid, -10, True, False, {}  # Invalid move (should not happen)

        # Check for win conditions
        winner = self.check_winner()
        if winner == self.current_player:
            self.done = True
            return self.grid, 1, True, False, {}

        # Switch to the other player
        self.current_player = 1 if self.current_player == 2 else 2

        return (
            self.grid,
            0,
            False,
            False,
            {},
        )  # Observation, reward, terminated, truncated, info

    def move_runner(self, player, direction):
        opponent = 1 if player == 2 else 2
        positions = np.argwhere(self.grid == player)
        if positions.size == 0:
            return False  # Runner not found

        r, c = positions[0]

        # Determine new position based on the direction
        if direction == 0:  # Up
            new_r, new_c = r - 1, c
        elif direction == 1:  # Down
            new_r, new_c = r + 1, c
        elif direction == 2:  # Left
            new_r, new_c = r, c - 1
        elif direction == 3:  # Right
            new_r, new_c = r, c + 1
        else:
            return False  # Invalid direction

        # Check if new position is within grid bounds
        if not (0 <= new_r < 5 and 0 <= new_c < 5):
            return False

        cell_value = self.grid[new_r, new_c]

        if cell_value == 0:
            # Move runner to the empty cell
            self.grid[r, c] = 0
            self.grid[new_r, new_c] = player
            return True
        elif cell_value == 3:
            # Barrier present
            return False
        elif cell_value == opponent:
            # Capture opponent's runner
            self.grid[r, c] = 0
            self.grid[new_r, new_c] = player
            self.done = True  # Player wins by capture
            return True
        else:
            # Cell occupied by own runner (should not happen)
            return False

    def place_barrier(self, player, row, col):
        if self.barriers[player] <= 0:
            return False  # No barriers left

        if not (0 <= row < 5 and 0 <= col < 5):
            return False  # Invalid position

        if self.grid[row, col] != 0:
            return False  # Cell is not empty

        # Cannot place barrier adjacent to own runner
        positions = np.argwhere(self.grid == player)
        if positions.size == 0:
            return False  # Runner not found

        r, c = positions[0]
        if abs(row - r) + abs(col - c) == 1:
            return False  # Cell is adjacent to own runner

        # Place the barrier
        self.grid[row, col] = 3
        self.barriers[player] -= 1
        return True

    def check_winner(self):
        positions = np.argwhere(self.grid == self.current_player)
        if positions.size == 0:
            return None  # Runner not found

        r, c = positions[0]

        # Check if the runner has reached the opponent's starting position
        if self.current_player == 1 and (r, c) == (4, 4):
            return 1
        elif self.current_player == 2 and (r, c) == (0, 0):
            return 2

        # The game continues
        return None

    def valid_moves(self):
        moves = []
        player = self.current_player
        opponent = 1 if player == 2 else 2
        positions = np.argwhere(self.grid == player)
        if positions.size == 0:
            return moves  # No valid moves if runner is not on the grid

        r, c = positions[0]

        # Movement actions (0: Up, 1: Down, 2: Left, 3: Right)
        for action in range(4):
            if action == 0:
                new_r, new_c = r - 1, c
            elif action == 1:
                new_r, new_c = r + 1, c
            elif action == 2:
                new_r, new_c = r, c - 1
            else:  # action == 3
                new_r, new_c = r, c + 1

            if 0 <= new_r < 5 and 0 <= new_c < 5:
                if self.grid[new_r, new_c] != 3:  # Not a barrier
                    moves.append(action)

        # Barrier placement actions
        if self.barriers[player] > 0:
            for cell_index in range(25):
                row = cell_index // 5
                col = cell_index % 5
                if self.grid[row, col] == 0:
                    if abs(row - r) + abs(col - c) != 1:
                        moves.append(4 + cell_index)

        return moves

    def render(self):
        grid_str = "  A B C D E\n"
        for i in range(5):
            row_str = str(i + 1) + " "
            for j in range(5):
                cell = self.grid[i][j]
                if cell == 0:
                    row_str += ". "
                elif cell == 1:
                    row_str += "1 "
                elif cell == 2:
                    row_str += "2 "
                elif cell == 3:
                    row_str += "X "
            grid_str += row_str + "\n"
        return grid_str
