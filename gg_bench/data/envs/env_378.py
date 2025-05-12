import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 4 move directions * 5 block options = 20 possible actions
        self.action_space = spaces.Discrete(20)

        # Observation space: 7x7 grid with values representing the state of each cell
        # 0: Empty, 1: Blocked, 2: Player 1 marker, 3: Player 2 marker
        self.observation_space = spaces.Box(low=0, high=3, shape=(7, 7), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid
        self.grid = np.zeros((7, 7), dtype=np.int8)

        # Set initial positions of the players
        self.grid[3, 0] = 2  # Player 1 marker
        self.grid[3, 6] = 3  # Player 2 marker

        # Store player positions
        self.player_positions = {1: (3, 0), 2: (3, 6)}

        # Set the current player (1 or 2)
        self.current_player = 1

        # Game state
        self.done = False

        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        # Decode the action into move and block directions
        move_direction = action // 5
        block_direction = (action % 5) - 1  # Adjust block_direction to -1 to 3

        # Map move and block directions to coordinate changes
        move_dict = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        block_dict = {-1: (0, 0), 0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        dx, dy = move_dict[move_direction]
        block_dx, block_dy = block_dict[block_direction]

        # Get current position
        pos = self.player_positions[self.current_player]
        new_x, new_y = pos[0] + dx, pos[1] + dy

        # Check move validity
        if not (0 <= new_x < 7 and 0 <= new_y < 7) or self.grid[new_x, new_y] != 0:
            self.done = True
            return self.grid.copy(), -10, True, False, {}

        # Move the marker
        self.grid[pos] = 0  # Remove marker from old position
        player_marker = 2 if self.current_player == 1 else 3
        self.grid[new_x, new_y] = player_marker  # Place marker in new position
        self.player_positions[self.current_player] = (new_x, new_y)  # Update position

        # Block placement
        if block_direction != -1:
            block_x, block_y = new_x + block_dx, new_y + block_dy
            # Check block placement validity
            if (
                0 <= block_x < 7
                and 0 <= block_y < 7
                and self.grid[block_x, block_y] == 0
            ):
                self.grid[block_x, block_y] = 1  # Place the block
            else:
                self.done = True
                return self.grid.copy(), -10, True, False, {}
        else:
            # Check if any valid block placements are available
            adjacent_cells = [
                (new_x + dx, new_y + dy)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ]
            block_available = any(
                0 <= x < 7 and 0 <= y < 7 and self.grid[x, y] == 0
                for x, y in adjacent_cells
            )
            if block_available:
                self.done = True
                return self.grid.copy(), -10, True, False, {}

        # Check if the opponent has any valid moves
        opponent = 2 if self.current_player == 1 else 1
        if not self.get_valid_moves_for_player(opponent):
            # Opponent has no valid moves; current player wins
            self.done = True
            return self.grid.copy(), 1, True, False, {}

        # Switch to the opponent
        self.current_player = opponent

        return self.grid.copy(), 0, False, False, {}

    def render(self):
        grid_str = "   " + " ".join(str(i) for i in range(7)) + "\n"
        for i in range(7):
            row = f"{i} "
            for j in range(7):
                value = self.grid[i, j]
                if value == 0:
                    cell = ". "
                elif value == 1:
                    cell = "X "
                elif value == 2:
                    cell = "P1"
                elif value == 3:
                    cell = "P2"
                else:
                    cell = "? "
                row += cell + " "
            grid_str += row + "\n"
        return grid_str

    def valid_moves(self):
        valid_actions = []
        pos = self.player_positions[self.current_player]
        move_dict = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        block_dict = {-1: (0, 0), 0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        for move_direction in range(4):
            dx, dy = move_dict[move_direction]
            new_x, new_y = pos[0] + dx, pos[1] + dy

            # Check move validity
            if 0 <= new_x < 7 and 0 <= new_y < 7 and self.grid[new_x, new_y] == 0:
                available_block_directions = []
                block_available = False

                # Check possible block placements
                for block_direction in range(-1, 4):
                    if block_direction == -1:
                        available_block_directions.append(block_direction)
                        continue
                    block_dx, block_dy = block_dict[block_direction]
                    block_x, block_y = new_x + block_dx, new_y + block_dy

                    if (
                        0 <= block_x < 7
                        and 0 <= block_y < 7
                        and self.grid[block_x, block_y] == 0
                    ):
                        block_available = True
                        available_block_directions.append(block_direction)

                # If no blocks can be placed, only include action with block_direction -1
                if not block_available:
                    available_block_directions = [-1]

                for block_direction in available_block_directions:
                    action = move_direction * 5 + (block_direction + 1)
                    valid_actions.append(action)

        return valid_actions

    def get_valid_moves_for_player(self, player):
        pos = self.player_positions[player]
        valid_moves = []
        move_dict = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        for move_direction in range(4):
            dx, dy = move_dict[move_direction]
            new_x, new_y = pos[0] + dx, pos[1] + dy
            if 0 <= new_x < 7 and 0 <= new_y < 7 and self.grid[new_x, new_y] == 0:
                valid_moves.append(move_direction)
        return valid_moves
