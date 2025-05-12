import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 32 possible actions (4 move directions * 8 block directions)
        self.action_space = spaces.Discrete(32)

        # Observation space: 4x4 grid with values:
        # Empty: 0, Blocked: -1, Player 1: 1, Player 2: 2
        self.observation_space = spaces.Box(low=-1, high=2, shape=(4, 4), dtype=np.int8)

        # Movement directions (up, down, left, right)
        self.move_directions = [
            (-1, 0),  # Up
            (1, 0),  # Down
            (0, -1),  # Left
            (0, 1),  # Right
        ]

        # Blocking directions (8 surrounding cells)
        self.block_directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),  # Upper row
            (0, -1),
            (0, 1),  # Middle row (excluding center)
            (1, -1),
            (1, 0),
            (1, 1),  # Lower row
        ]

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((4, 4), dtype=np.int8)
        self.grid[0, 0] = 1  # Player 1 starts at (0,0)
        self.grid[3, 3] = 2  # Player 2 starts at (3,3)
        self.current_player = 1
        self.done = False
        self.info = {}
        return self.grid.copy(), self.info  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                self.grid.copy(),
                0,
                True,
                False,
                self.info,
            )  # Game is already over

        # Decode action
        move_direction_index = action // 8
        block_direction_index = action % 8

        if (
            move_direction_index < 0
            or move_direction_index >= 4
            or block_direction_index < 0
            or block_direction_index >= 8
        ):
            # Invalid action index
            self.done = True
            return (
                self.grid.copy(),
                -10,
                True,
                False,
                self.info,
            )

        move_direction = self.move_directions[move_direction_index]
        block_direction = self.block_directions[block_direction_index]

        # Find current player's position
        player_pos = np.argwhere(self.grid == self.current_player)
        if player_pos.size == 0:
            # Player's token not found
            self.done = True
            return (
                self.grid.copy(),
                -10,
                True,
                False,
                self.info,
            )
        player_pos = player_pos[0]  # Get the first occurrence

        # Calculate new position after move
        new_pos = player_pos + move_direction
        new_row, new_col = new_pos

        # Check move validity
        if (
            new_row < 0
            or new_row >= 4
            or new_col < 0
            or new_col >= 4
            or self.grid[new_row, new_col] != 0
        ):
            # Invalid move
            self.done = True
            return (
                self.grid.copy(),
                -10,
                True,
                False,
                self.info,
            )

        # Perform the move
        self.grid[player_pos[0], player_pos[1]] = 0  # Remove from old position
        self.grid[new_row, new_col] = self.current_player  # Place at new position

        # Check for win condition (reached opponent's starting position)
        opponent_start_pos = (3, 3) if self.current_player == 1 else (0, 0)
        if (new_row, new_col) == opponent_start_pos:
            self.done = True
            return self.grid.copy(), 1, True, False, self.info  # Current player wins

        # Block a cell
        block_pos = new_pos + block_direction
        block_row, block_col = block_pos

        if (
            0 <= block_row < 4
            and 0 <= block_col < 4
            and self.grid[block_row, block_col] == 0
        ):
            self.grid[block_row, block_col] = -1  # Block the cell
        else:
            # Invalid block (cell is out of bounds or not empty)
            # Proceed without blocking
            pass  # According to rules, no penalty for being unable to block

        # Check if opponent has any valid moves
        opponent = 2 if self.current_player == 1 else 1
        opponent_pos = np.argwhere(self.grid == opponent)
        if opponent_pos.size == 0:
            # Opponent's token not found
            self.done = True
            return (
                self.grid.copy(),
                1,  # Current player wins
                True,
                False,
                self.info,
            )
        opponent_pos = opponent_pos[0]

        opponent_moves = self.get_valid_moves(opponent_pos)
        if len(opponent_moves) == 0:
            # Opponent cannot move, current player wins
            self.done = True
            return self.grid.copy(), 1, True, False, self.info  # Current player wins

        # Switch to the next player
        self.current_player = opponent

        return self.grid.copy(), 0, False, False, self.info  # Continue game

    def get_valid_moves(self, position):
        valid_moves = []
        for direction in self.move_directions:
            new_pos = position + direction
            new_row, new_col = new_pos
            if (
                0 <= new_row < 4
                and 0 <= new_col < 4
                and self.grid[new_row, new_col] == 0
            ):
                valid_moves.append((new_row, new_col))
        return valid_moves

    def render(self):
        symbols = {
            0: ".",
            -1: "X",
            1: "P1",
            2: "P2",
        }
        grid_str = ""
        for row in range(4):
            for col in range(4):
                cell = self.grid[row, col]
                grid_str += f"{symbols[cell]:>3} "
            grid_str += "\n"
        return grid_str

    def valid_moves(self):
        if self.done:
            return []

        # Find current player's position
        player_pos = np.argwhere(self.grid == self.current_player)
        if player_pos.size == 0:
            return []

        player_pos = player_pos[0]

        valid_actions = []
        for move_idx, move_direction in enumerate(self.move_directions):
            # Calculate new position after move
            new_pos = player_pos + move_direction
            new_row, new_col = new_pos
            if (
                0 <= new_row < 4
                and 0 <= new_col < 4
                and self.grid[new_row, new_col] == 0
            ):
                # For each valid move, check valid blocks
                for block_idx, block_direction in enumerate(self.block_directions):
                    block_pos = new_pos + block_direction
                    block_row, block_col = block_pos
                    if (
                        0 <= block_row < 4
                        and 0 <= block_col < 4
                        and self.grid[block_row, block_col] == 0
                    ):
                        action = move_idx * 8 + block_idx
                        valid_actions.append(action)
                # If no valid blocks, still need to add action with block skipped
                if not any(
                    0 <= (new_pos + d)[0] < 4
                    and 0 <= (new_pos + d)[1] < 4
                    and self.grid[(new_pos + d)[0], (new_pos + d)[1]] == 0
                    for d in self.block_directions
                ):
                    # Artificially choose block_idx = 0 when no blocks are possible
                    action = move_idx * 8 + 0
                    valid_actions.append(action)
        return valid_actions
