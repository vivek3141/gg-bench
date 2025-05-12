import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Each token has up to 37 possible actions
        # Total actions = 5 tokens * 37 actions = 185
        self.action_space = spaces.Discrete(185)

        # Observation space: Flattened array containing grid state, block durations,
        # specials used, and current player
        # Grid state: 25 cells
        # Block durations: 25 cells
        # Specials used: 10 tokens
        # Current player: 1 integer
        # Total observation size = 25 + 25 + 10 + 1 = 61
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(61,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state variables
        # Grid state: 5x5 grid
        self.grid = np.zeros((5, 5), dtype=np.int32)

        # Block durations: 5x5 grid
        self.block_durations = np.zeros((5, 5), dtype=np.int32)

        # Place Player 1 tokens (IDs 1-5) at row 0
        for col in range(5):
            self.grid[0, col] = col + 1  # Tokens A1-A5 IDs 1-5

        # Place Player 2 tokens (IDs 6-10) at row 4
        for col in range(5):
            self.grid[4, col] = col + 6  # Tokens B1-B5 IDs 6-10

        # Tokens' special commands usage: 0 (not used) or 1 (used)
        self.specials_used = np.zeros(
            10, dtype=np.int32
        )  # Index 0-9 for tokens IDs 1-10

        # Initialize scores
        self.scores = {1: 0, 2: 0}

        # Current player: 1 or 2
        self.current_player = 1

        # Track alive tokens for each player
        self.tokens_alive = {
            1: [1, 2, 3, 4, 5],  # Player 1 tokens IDs
            2: [6, 7, 8, 9, 10],  # Player 2 tokens IDs
        }

        # Map token IDs to indices in specials_used array
        self.token_id_to_index = {i + 1: i for i in range(10)}  # Token IDs 1-10

        self.done = False

        # Return the initial observation and info
        return self.get_observation(), {}

    def get_observation(self):
        # Flatten the grid, block durations, specials used, and current player
        grid_flat = self.grid.flatten()
        blocks_flat = self.block_durations.flatten()
        specials_flat = self.specials_used
        current_player = np.array([self.current_player], dtype=np.int32)
        observation = np.concatenate(
            [grid_flat, blocks_flat, specials_flat, current_player]
        )
        return observation

    def is_within_grid(self, row, col):
        return 0 <= row < 5 and 0 <= col < 5

    def update_blocks(self):
        # Reduce block durations by 1
        self.block_durations[self.block_durations > 0] -= 1

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, True, False, {}

        # Map action index to token and sub-action index
        token_index = action // 37  # Integer division to get the token index (0-4)
        sub_action_index = action % 37  # Remainder to get the sub-action index

        # Get the token ID based on the current player
        if self.current_player == 1:
            token_id = token_index + 1  # Tokens A1-A5 IDs 1-5
        else:
            token_id = token_index + 6  # Tokens B1-B5 IDs 6-10

        # Check if the token is alive
        if token_id not in self.tokens_alive[self.current_player]:
            # Invalid move: token is not alive
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Get the token's current position
        token_pos = np.argwhere(self.grid == token_id)
        if token_pos.shape[0] == 0:
            # Invalid move: token is not on the grid
            self.done = True
            return self.get_observation(), -10, True, False, {}

        row, col = token_pos[0]
        gained_points = 0
        valid = True

        if sub_action_index < 4:
            # Movement action
            direction = sub_action_index
            if direction == 0:
                new_row, new_col = row - 1, col  # Up
            elif direction == 1:
                new_row, new_col = row + 1, col  # Down
            elif direction == 2:
                new_row, new_col = row, col - 1  # Left
            elif direction == 3:
                new_row, new_col = row, col + 1  # Right
            else:
                valid = False

            if not self.is_within_grid(new_row, new_col):
                valid = False
            else:
                dest_cell = self.grid[new_row, new_col]
                if self.block_durations[new_row, new_col] > 0:
                    valid = False
                else:
                    if dest_cell == 0:
                        # Move to empty cell
                        self.grid[row, col] = 0
                        self.grid[new_row, new_col] = token_id
                        gained_points += 1
                    elif dest_cell >= 1 and dest_cell <= 10:
                        # Check if opponent's token
                        if (
                            dest_cell in self.tokens_alive[1]
                            and self.current_player == 1
                        ) or (
                            dest_cell in self.tokens_alive[2]
                            and self.current_player == 2
                        ):
                            valid = False
                        else:
                            # Capture opponent's token
                            self.grid[row, col] = 0
                            self.grid[new_row, new_col] = token_id
                            opponent = 1 if self.current_player == 2 else 2
                            self.tokens_alive[opponent].remove(dest_cell)
                            gained_points += 2
                    else:
                        valid = False
        elif sub_action_index >= 4 and sub_action_index < 29:
            # Special command: Jump
            token_special_index = self.token_id_to_index[token_id]
            if self.specials_used[token_special_index]:
                valid = False
            else:
                cell_index = sub_action_index - 4  # Cell index from 0 to 24
                dest_row = cell_index // 5
                dest_col = cell_index % 5
                manhattan_distance = abs(dest_row - row) + abs(dest_col - col)
                if manhattan_distance > 2:
                    valid = False
                else:
                    dest_cell = self.grid[dest_row, dest_col]
                    if dest_cell == 0 and self.block_durations[dest_row, dest_col] == 0:
                        # Jump to empty cell
                        self.grid[row, col] = 0
                        self.grid[dest_row, dest_col] = token_id
                        gained_points += 1  # Captured cell
                        self.specials_used[token_special_index] = 1
                        gained_points += 1  # Special command bonus
                    elif dest_cell >= 1 and dest_cell <= 10:
                        # Check if opponent's token
                        if (
                            dest_cell in self.tokens_alive[1]
                            and self.current_player == 1
                        ) or (
                            dest_cell in self.tokens_alive[2]
                            and self.current_player == 2
                        ):
                            valid = False
                        else:
                            # Capture opponent's token
                            self.grid[row, col] = 0
                            self.grid[dest_row, dest_col] = token_id
                            opponent = 1 if self.current_player == 2 else 2
                            self.tokens_alive[opponent].remove(dest_cell)
                            gained_points += 2  # Captured token
                            self.specials_used[token_special_index] = 1
                            gained_points += 1  # Special command bonus
                    else:
                        valid = False
        elif sub_action_index >= 29 and sub_action_index < 33:
            # Special command: Swap
            token_special_index = self.token_id_to_index[token_id]
            if self.specials_used[token_special_index]:
                valid = False
            else:
                swap_token_idx = sub_action_index - 29
                swap_token_id = None
                if self.current_player == 1:
                    swap_token_id = swap_token_idx + 1  # Tokens IDs 1-5
                else:
                    swap_token_id = swap_token_idx + 6  # Tokens IDs 6-10
                if swap_token_id == token_id:
                    valid = False
                elif swap_token_id not in self.tokens_alive[self.current_player]:
                    valid = False
                else:
                    swap_pos = np.argwhere(self.grid == swap_token_id)
                    if swap_pos.shape[0] == 0:
                        valid = False
                    else:
                        swap_row, swap_col = swap_pos[0]
                        # Swap positions
                        self.grid[row, col], self.grid[swap_row, swap_col] = (
                            self.grid[swap_row, swap_col],
                            self.grid[row, col],
                        )
                        self.specials_used[token_special_index] = 1
                        gained_points += 1  # Special command bonus
        elif sub_action_index >= 33 and sub_action_index < 37:
            # Special command: Block
            token_special_index = self.token_id_to_index[token_id]
            if self.specials_used[token_special_index]:
                valid = False
            else:
                direction = sub_action_index - 33
                if direction == 0:
                    block_row, block_col = row - 1, col  # Up
                elif direction == 1:
                    block_row, block_col = row + 1, col  # Down
                elif direction == 2:
                    block_row, block_col = row, col - 1  # Left
                elif direction == 3:
                    block_row, block_col = row, col + 1  # Right
                else:
                    valid = False

                if not self.is_within_grid(block_row, block_col):
                    valid = False
                else:
                    if (
                        self.grid[block_row, block_col] == 0
                        and self.block_durations[block_row, block_col] == 0
                    ):
                        # Place block
                        self.block_durations[block_row, block_col] = 2
                        self.specials_used[token_special_index] = 1
                        gained_points += 1  # Special command bonus
                    else:
                        valid = False
        else:
            valid = False

        if not valid:
            # Invalid move
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Update the player's score
        self.scores[self.current_player] += gained_points

        # Update block durations
        self.update_blocks()

        # Check for win condition
        if self.scores[self.current_player] >= 10:
            self.done = True
            return self.get_observation(), 1, True, False, {}

        # Switch to the other player
        self.current_player = 1 if self.current_player == 2 else 2

        # Return the new observation and info
        return self.get_observation(), 0, False, False, {}

    def render(self):
        # Return a string representation of the grid
        grid_repr = ""
        for r in range(5):
            row_str = ""
            for c in range(5):
                cell = self.grid[r, c]
                block = self.block_durations[r, c]
                if cell == 0:
                    if block > 0:
                        row_str += " X "  # Blocked cell
                    else:
                        row_str += " . "  # Empty cell
                elif cell >= 1 and cell <= 5:
                    row_str += f" A{cell} "  # Player 1's token
                elif cell >= 6 and cell <= 10:
                    row_str += f" B{cell - 5} "  # Player 2's token
                else:
                    row_str += " ? "  # Unknown cell
            grid_repr += row_str + "\n"
        return grid_repr

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        # Get token indices for the current player
        if self.current_player == 1:
            token_ids = self.tokens_alive[1]
            token_indices = [tid - 1 for tid in token_ids]  # Tokens A1-A5 IDs 1-5
        else:
            token_ids = self.tokens_alive[2]
            token_indices = [tid - 6 for tid in token_ids]  # Tokens B1-B5 IDs 6-10

        for token_index in token_indices:
            token_id = token_index + (1 if self.current_player == 1 else 6)
            # Get the token's position
            token_pos = np.argwhere(self.grid == token_id)
            if token_pos.shape[0] == 0:
                continue  # Token not on grid
            row, col = token_pos[0]
            # Movement actions
            for sub_action_index in range(0, 4):
                direction = sub_action_index
                if direction == 0:
                    new_row, new_col = row - 1, col  # Up
                elif direction == 1:
                    new_row, new_col = row + 1, col  # Down
                elif direction == 2:
                    new_row, new_col = row, col - 1  # Left
                elif direction == 3:
                    new_row, new_col = row, col + 1  # Right
                if not self.is_within_grid(new_row, new_col):
                    continue
                dest_cell = self.grid[new_row, new_col]
                if self.block_durations[new_row, new_col] > 0:
                    continue
                if dest_cell == 0 or (
                    (dest_cell in self.tokens_alive[1] and self.current_player == 2)
                    or (dest_cell in self.tokens_alive[2] and self.current_player == 1)
                ):
                    action = token_index * 37 + sub_action_index
                    valid_actions.append(action)
            # Special commands
            token_special_index = self.token_id_to_index[token_id]
            if self.specials_used[token_special_index] == 0:
                # Jump actions
                for cell_index in range(25):
                    dest_row = cell_index // 5
                    dest_col = cell_index % 5
                    manhattan_distance = abs(dest_row - row) + abs(dest_col - col)
                    if manhattan_distance > 2:
                        continue
                    dest_cell = self.grid[dest_row, dest_col]
                    if dest_cell == 0 and self.block_durations[dest_row, dest_col] == 0:
                        action = token_index * 37 + (4 + cell_index)
                        valid_actions.append(action)
                    elif (
                        dest_cell in self.tokens_alive[1] and self.current_player == 2
                    ) or (
                        dest_cell in self.tokens_alive[2] and self.current_player == 1
                    ):
                        action = token_index * 37 + (4 + cell_index)
                        valid_actions.append(action)
                # Swap actions
                swap_token_indices = [
                    i
                    for i in range(5)
                    if i != token_index
                    and (
                        (i + 1 in self.tokens_alive[1])
                        if self.current_player == 1
                        else (i + 6 in self.tokens_alive[2])
                    )
                ]
                for swap_idx in swap_token_indices:
                    action = token_index * 37 + (29 + swap_idx)
                    valid_actions.append(action)
                # Block actions
                for dir_idx in range(4):
                    if dir_idx == 0:
                        block_row, block_col = row - 1, col  # Up
                    elif dir_idx == 1:
                        block_row, block_col = row + 1, col  # Down
                    elif dir_idx == 2:
                        block_row, block_col = row, col - 1  # Left
                    elif dir_idx == 3:
                        block_row, block_col = row, col + 1  # Right
                    if not self.is_within_grid(block_row, block_col):
                        continue
                    if (
                        self.grid[block_row, block_col] == 0
                        and self.block_durations[block_row, block_col] == 0
                    ):
                        action = token_index * 37 + (33 + dir_idx)
                        valid_actions.append(action)
        return valid_actions
