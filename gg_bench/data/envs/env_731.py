import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 4 movement directions * 5 lock placements (including skip)
        self.action_space = spaces.Discrete(20)

        # Observation space: 10x10 grid, values 0 (empty), 1 (Player 1 cursor), 2 (Player 2 cursor), 3 (lock)
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(10, 10), dtype=np.uint8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((10, 10), dtype=np.uint8)
        self.grid[0, 0] = 1  # Player 1 cursor at (0,0)
        self.grid[9, 9] = 2  # Player 2 cursor at (9,9)
        self.player_positions = {1: (0, 0), 2: (9, 9)}
        self.current_player = 1
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        move_direction = action // 5
        lock_direction = action % 5  # 0-3: directions, 4: skip

        # Map move direction to delta row,col
        move_deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        lock_deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        curr_row, curr_col = self.player_positions[self.current_player]

        # Movement Phase
        delta_row, delta_col = move_deltas.get(move_direction, (0, 0))
        new_row = curr_row + delta_row
        new_col = curr_col + delta_col

        # Check grid boundaries
        if not (0 <= new_row < 10 and 0 <= new_col < 10):
            return self.grid.copy(), -10, True, False, {}  # Invalid move

        # Check if new cell is empty
        if self.grid[new_row, new_col] != 0:
            return self.grid.copy(), -10, True, False, {}  # Invalid move

        # Move the cursor
        self.grid[curr_row, curr_col] = 0  # Remove cursor from old position
        self.grid[new_row, new_col] = (
            self.current_player
        )  # Place cursor at new position
        self.player_positions[self.current_player] = (new_row, new_col)

        opponent_start_pos = {1: (9, 9), 2: (0, 0)}

        # Check for victory by reaching opponent's starting corner
        if (new_row, new_col) == opponent_start_pos[self.current_player]:
            self.done = True
            return self.grid.copy(), 1, True, False, {}  # Current player wins

        # Lock Placement Phase
        valid_lock_cells = []
        for direction, (dr, dc) in lock_deltas.items():
            adj_row = new_row + dr
            adj_col = new_col + dc
            if 0 <= adj_row < 10 and 0 <= adj_col < 10:
                if self.grid[adj_row, adj_col] == 0:
                    valid_lock_cells.append(direction)
        if len(valid_lock_cells) == 0:
            if lock_direction != 4:
                return (
                    self.grid.copy(),
                    -10,
                    True,
                    False,
                    {},
                )  # Invalid lock placement (should skip)
            # Skip lock placement
        else:
            if lock_direction == 4:
                return (
                    self.grid.copy(),
                    -10,
                    True,
                    False,
                    {},
                )  # Invalid lock placement (should place a lock)
            elif lock_direction in valid_lock_cells:
                dr, dc = lock_deltas[lock_direction]
                lock_row = new_row + dr
                lock_col = new_col + dc
                self.grid[lock_row, lock_col] = 3  # Place lock
            else:
                return self.grid.copy(), -10, True, False, {}  # Invalid lock placement

        # Check for victory by immobilizing opponent
        opponent = 2 if self.current_player == 1 else 1
        opp_row, opp_col = self.player_positions[opponent]
        move_deltas_list = list(move_deltas.values())
        opp_moves = []
        for dr, dc in move_deltas_list:
            adj_row = opp_row + dr
            adj_col = opp_col + dc
            if 0 <= adj_row < 10 and 0 <= adj_col < 10:
                if self.grid[adj_row, adj_col] == 0:
                    opp_moves.append((adj_row, adj_col))
        if len(opp_moves) == 0:
            self.done = True
            return (
                self.grid.copy(),
                1,
                True,
                False,
                {},
            )  # Current player wins (immobilized opponent)

        # Switch to next player
        self.current_player = opponent
        return self.grid.copy(), 0, False, False, {}  # Continue game

    def render(self):
        grid_str = "    " + " ".join([str(i) for i in range(10)]) + "\n"
        grid_str += "   " + "-" * 21 + "\n"
        for i in range(10):
            row_str = str(i) + " |"
            for j in range(10):
                cell = self.grid[i, j]
                if cell == 0:
                    row_str += " ."
                elif cell == 1:
                    row_str += " P1"
                elif cell == 2:
                    row_str += " P2"
                elif cell == 3:
                    row_str += " L"
            row_str += "\n"
            grid_str += row_str
        grid_str += "   " + "-" * 21 + "\n"
        return grid_str

    def valid_moves(self):
        valid_actions = []
        move_deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }
        lock_deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        curr_row, curr_col = self.player_positions[self.current_player]

        # Find valid movement directions
        valid_moves_dirs = []
        for move_dir, (dr, dc) in move_deltas.items():
            new_row = curr_row + dr
            new_col = curr_col + dc
            if 0 <= new_row < 10 and 0 <= new_col < 10:
                if self.grid[new_row, new_col] == 0:
                    valid_moves_dirs.append(move_dir)
        for move_dir in valid_moves_dirs:
            # For each valid move, find valid lock placements
            dr_move, dc_move = move_deltas[move_dir]
            new_row = curr_row + dr_move
            new_col = curr_col + dc_move
            valid_lock_dirs = []
            for lock_dir, (dr_lock, dc_lock) in lock_deltas.items():
                lock_row = new_row + dr_lock
                lock_col = new_col + dc_lock
                if 0 <= lock_row < 10 and 0 <= lock_col < 10:
                    if self.grid[lock_row, lock_col] == 0:
                        valid_lock_dirs.append(lock_dir)
            if len(valid_lock_dirs) == 0:
                # Can skip lock placement
                action = move_dir * 5 + 4
                valid_actions.append(action)
            else:
                for lock_dir in valid_lock_dirs:
                    action = move_dir * 5 + lock_dir
                    valid_actions.append(action)
        return valid_actions
