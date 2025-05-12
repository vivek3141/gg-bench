import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions:
        #   0: Move Up
        #   1: Move Down
        #   2: Move Left
        #   3: Move Right
        #   4-27: Place Block at positions excluding the portal cell (positions 0 to 24, skipping index 12)
        self.action_space = spaces.Discrete(28)

        # Observation space: 5x5 grid with values:
        #   0: Empty
        #   1: Player 1
        #   2: Player 2
        #   3: Portal
        #   4: Block
        self.observation_space = spaces.Box(low=0, high=4, shape=(5, 5), dtype=np.int8)

        self.portal_position = (2, 2)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.player_positions = {1: (0, 0), 2: (4, 4)}
        self.grid[self.player_positions[1]] = 1
        self.grid[self.player_positions[2]] = 2
        self.grid[self.portal_position] = 3  # Portal
        self.portal_passed = {1: False, 2: False}
        self.current_player = 1
        self.done = False
        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            # Forfeit turn
            self.current_player = 3 - self.current_player
            return self.grid.copy(), reward, self.done, False, {}

        # Execute the action
        if action in [0, 1, 2, 3]:
            # Movement action
            reward = -10  # Default reward for valid move
            direction = action
            # Get current position
            pos = self.player_positions[self.current_player]
            if direction == 0:  # Up
                new_pos = (pos[0] - 1, pos[1])
            elif direction == 1:  # Down
                new_pos = (pos[0] + 1, pos[1])
            elif direction == 2:  # Left
                new_pos = (pos[0], pos[1] - 1)
            elif direction == 3:  # Right
                new_pos = (pos[0], pos[1] + 1)

            # Move player
            self.grid[pos] = 0  # Clear old position
            self.player_positions[self.current_player] = new_pos
            if self.grid[new_pos] == 3:
                # Player passes through the portal
                self.portal_passed[self.current_player] = True
            self.grid[new_pos] = self.current_player  # Set new position

            # Check for win condition
            opponent_start_pos = (4, 4) if self.current_player == 1 else (0, 0)
            if (
                new_pos == opponent_start_pos
                and self.portal_passed[self.current_player]
            ):
                reward = 1
                self.done = True
            else:
                # Check for stalemate condition
                opponent_valid_actions = self.valid_moves(
                    player=3 - self.current_player
                )
                if not opponent_valid_actions:
                    reward = 1
                    self.done = True

        else:
            # Block placement action
            reward = -10  # Default reward for valid move
            grid_index = action - 4
            if grid_index >= 12:
                grid_index += 1  # Adjust for portal cell
            row = grid_index // 5
            col = grid_index % 5
            block_pos = (row, col)
            self.grid[block_pos] = 4  # Place block

            # Check for stalemate condition
            opponent_valid_actions = self.valid_moves(player=3 - self.current_player)
            if not opponent_valid_actions:
                reward = 1
                self.done = True

        if not self.done:
            self.current_player = 3 - self.current_player  # Switch player

        return self.grid.copy(), reward, self.done, False, {}

    def valid_moves(self, player=None):
        if player is None:
            player = self.current_player
        valid_actions = []

        # Movement actions
        directions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        pos = self.player_positions[player]
        for action in [0, 1, 2, 3]:
            dir = directions[action]
            new_row = pos[0] + dir[0]
            new_col = pos[1] + dir[1]
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                cell_content = self.grid[new_row, new_col]
                if cell_content in [0, 3]:  # Empty or portal
                    valid_actions.append(action)

        # Block placement actions
        opponent = 3 - player
        opponent_pos = self.player_positions[opponent]
        opponent_adjacent_positions = []
        for dir in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = opponent_pos[0] + dir[0], opponent_pos[1] + dir[1]
            if 0 <= r < 5 and 0 <= c < 5:
                opponent_adjacent_positions.append((r, c))

        for row in range(5):
            for col in range(5):
                block_pos = (row, col)
                if block_pos == self.portal_position:
                    continue  # Cannot place block on portal cell
                if self.grid[block_pos] != 0:
                    continue  # Only place on empty cells
                if (
                    block_pos == self.player_positions[1]
                    or block_pos == self.player_positions[2]
                ):
                    continue  # Cannot place on players' positions
                if block_pos in opponent_adjacent_positions:
                    continue  # Cannot place adjacent to opponent's current position

                # Map grid position to action index
                grid_index = row * 5 + col
                if grid_index >= 12:
                    grid_index -= 1  # Adjust for portal cell
                action = grid_index + 4
                valid_actions.append(action)

        return valid_actions

    def render(self):
        cell_symbols = {0: "  ", 1: "P1", 2: "P2", 3: "O ", 4: "X "}
        grid_str = ""
        for row in range(5):
            grid_str += "+----" * 5 + "+\n"
            for col in range(5):
                content = self.grid[row, col]
                symbol = cell_symbols[content]
                grid_str += f"| {symbol}"
            grid_str += "|\n"
        grid_str += "+----" * 5 + "+\n"
        return grid_str
