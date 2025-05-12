import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)  # Actions 0-8 correspond to numbers 1-9

        # Observation space:
        # Positions 0-8: Grid state (-1: claimed by Player 2, 0: unclaimed, 1: claimed by Player 1)
        # Position 9: Last number selected by the opponent (0 if no number selected yet)
        # Position 10: Current player indicator (1 or -1)
        self.observation_space = spaces.Box(low=-1, high=9, shape=(11,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros(9, dtype=np.int8)  # Grid positions are unclaimed
        self.last_number_selected = 0  # No number selected yet
        self.current_player = 1  # Player 1 starts
        self.first_move = True
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Return the current observation: grid state, last number selected, current player
        obs = np.append(self.grid, [self.last_number_selected, self.current_player])
        return obs

    def step(self, action):
        action = int(action)
        number_selected = action + 1  # Map action index to number on the grid (1-9)

        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game already over

        if self.grid[action] != 0:
            # Invalid move: position already claimed
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Validate the move according to the game rules
        if self.first_move:
            valid_move = True
        else:
            valid_move = self._is_valid_move(number_selected, self.last_number_selected)

        if not valid_move:
            # Invalid move according to game rules
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid move: update the grid and game state
        self.grid[action] = self.current_player
        self.last_number_selected = number_selected
        self.first_move = False

        # Check if the opponent has any valid moves
        opponent_valid_moves = self.valid_moves()
        if len(opponent_valid_moves) == 0:
            # Opponent has no valid moves: current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch to the opponent
        self.current_player *= -1
        return self._get_obs(), 0, False, False, {}

    def _is_valid_move(self, number_selected, last_number_selected):
        # Check if the selected number is a divisor or multiple of the last number
        if number_selected == last_number_selected:
            return True
        if (
            number_selected % last_number_selected == 0
            or last_number_selected % number_selected == 0
        ):
            return True
        return False

    def valid_moves(self):
        # Return a list of valid action indices for the current player
        valid_actions = []
        last_number_selected = self.last_number_selected

        for action in range(9):
            if self.grid[action] == 0:
                number = action + 1
                if self.first_move:
                    valid_actions.append(action)
                else:
                    if self._is_valid_move(number, last_number_selected):
                        valid_actions.append(action)
        return valid_actions

    def render(self):
        # Provide a visual representation of the current game state
        grid_display = ""
        for i in range(3):
            row = ""
            for j in range(3):
                idx = i * 3 + j
                cell = self.grid[idx]
                if cell == 1:
                    row += " X "
                elif cell == -1:
                    row += " O "
                else:
                    row += f" {idx + 1} "
                if j < 2:
                    row += "|"
            grid_display += row
            if i < 2:
                grid_display += "\n-----------\n"
            else:
                grid_display += "\n"
        return grid_display
