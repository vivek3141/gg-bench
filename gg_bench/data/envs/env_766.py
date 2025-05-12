import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0-Up, 1-Down, 2-Left, 3-Right
        self.action_space = spaces.Discrete(4)

        # Define observation space
        # Indices 0-15: Grid numbers (1-4)
        # Index 16: Player 1 position (0-15)
        # Index 17: Player 2 position (0-15)
        # Index 18: Current player (1 or 2)
        low = np.array([1] * 16 + [0, 0, 1], dtype=np.int8)
        high = np.array([4] * 16 + [15, 15, 2], dtype=np.int8)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid with random numbers from 1 to 4
        self.grid_numbers = self.np_random.integers(1, 5, size=(4, 4), dtype=np.int8)

        # Player positions (flat indices from 0 to 15)
        self.P1_position = 0  # Cell (0, 0)
        self.P2_position = 15  # Cell (3, 3)

        # Set the current player: 1 for Player 1, 2 for Player 2
        self.current_player = 1

        self.done = False
        self.info = {}

        return self._get_observation(), self.info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, False, self.info

        # Validate action
        if action not in [0, 1, 2, 3]:
            return self._get_observation(), -10, True, False, self.info

        # Get current player position
        current_pos = self.P1_position if self.current_player == 1 else self.P2_position
        opponent_pos = (
            self.P2_position if self.current_player == 1 else self.P1_position
        )
        opponent_start_pos = (
            15 if self.current_player == 1 else 0
        )  # Opponent's starting cell

        row, col = divmod(current_pos, 4)
        number_on_cell = self.grid_numbers[row, col]

        # Determine the target position based on action
        dr, dc = 0, 0
        if action == 0:  # Up
            dr = -number_on_cell
        elif action == 1:  # Down
            dr = number_on_cell
        elif action == 2:  # Left
            dc = -number_on_cell
        elif action == 3:  # Right
            dc = number_on_cell

        new_row = row + dr
        new_col = col + dc

        # Check for out-of-bounds move
        if not (0 <= new_row < 4 and 0 <= new_col < 4):
            # Invalid move; player loses
            self.done = True
            return self._get_observation(), -10, True, False, self.info

        new_pos = new_row * 4 + new_col

        # Check if landing on opponent's current position
        if new_pos == opponent_pos:
            # Invalid move; player loses
            self.done = True
            return self._get_observation(), -10, True, False, self.info

        # Move is valid; update position
        if self.current_player == 1:
            self.P1_position = new_pos
        else:
            self.P2_position = new_pos

        # Check for win condition
        if new_pos == opponent_start_pos:
            # Current player wins
            self.done = True
            return self._get_observation(), 1, True, False, self.info

        # Check if player has any valid moves on next turn
        if not self._has_valid_moves(opponent=True):
            # Opponent cannot move; current player wins
            self.done = True
            return self._get_observation(), 1, True, False, self.info

        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1

        return self._get_observation(), 0, False, False, self.info

    def render(self):
        grid_display = ""
        for i in range(4):
            grid_display += "+-----" * 4 + "+\n"
            for j in range(4):
                cell_index = i * 4 + j
                cell_number = self.grid_numbers[i, j]
                player_here = ""
                if self.P1_position == cell_index:
                    player_here += "P1"
                if self.P2_position == cell_index:
                    player_here += "P2"
                if player_here:
                    content = f"{player_here}"
                else:
                    content = f" {cell_number} "
                grid_display += f"| {content:^3} "
            grid_display += "|\n"
        grid_display += "+-----" * 4 + "+\n"
        return grid_display

    def valid_moves(self):
        valid_actions = []
        current_pos = self.P1_position if self.current_player == 1 else self.P2_position
        opponent_pos = (
            self.P2_position if self.current_player == 1 else self.P1_position
        )
        row, col = divmod(current_pos, 4)
        number_on_cell = self.grid_numbers[row, col]

        for action in range(4):
            dr, dc = 0, 0
            if action == 0:  # Up
                dr = -number_on_cell
            elif action == 1:  # Down
                dr = number_on_cell
            elif action == 2:  # Left
                dc = -number_on_cell
            elif action == 3:  # Right
                dc = number_on_cell

            new_row = row + dr
            new_col = col + dc

            if 0 <= new_row < 4 and 0 <= new_col < 4:
                new_pos = new_row * 4 + new_col
                if new_pos != opponent_pos:
                    valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        observation = np.concatenate(
            [
                self.grid_numbers.flatten(),
                np.array(
                    [self.P1_position, self.P2_position, self.current_player],
                    dtype=np.int8,
                ),
            ]
        ).astype(np.int8)
        return observation

    def _has_valid_moves(self, opponent=False):
        current_pos = (
            self.P2_position
            if (self.current_player == 1) ^ opponent
            else self.P1_position
        )
        opponent_pos = (
            self.P1_position
            if (self.current_player == 1) ^ opponent
            else self.P2_position
        )
        row, col = divmod(current_pos, 4)
        number_on_cell = self.grid_numbers[row, col]

        for action in range(4):
            dr, dc = 0, 0
            if action == 0:  # Up
                dr = -number_on_cell
            elif action == 1:  # Down
                dr = number_on_cell
            elif action == 2:  # Left
                dc = -number_on_cell
            elif action == 3:  # Right
                dc = number_on_cell

            new_row = row + dr
            new_col = col + dc

            if 0 <= new_row < 4 and 0 <= new_col < 4:
                new_pos = new_row * 4 + new_col
                if new_pos != opponent_pos:
                    return True
        return False
