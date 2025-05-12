import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Two types of actions: Scan and Dig, at 25 locations
        # Total actions: 50
        self.action_space = spaces.Discrete(50)

        # Observation space: 25 grid cells + scans_remaining
        # Each grid cell can have values from -3 to 8
        # -3: dug and found treasure (should be terminal state)
        # -2: dug and no treasure
        # -1: untouched
        # 0-8: scanned, with distance value
        # Scans_remaining can be from 0 to 3
        # So total observation shape: (26,)
        self.observation_space = spaces.Box(low=-3, high=8, shape=(26,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the game state

        # Randomly assign treasure locations for both players
        self.treasure_locations = [None, None]  # index 0 and 1 for players

        # Initialize the grid observations for both players
        self.player_grids = [
            np.full(25, -1, dtype=np.int32),
            np.full(25, -1, dtype=np.int32),
        ]  # -1 indicates unknown

        # Scans remaining for both players
        self.scans_remaining = [3, 3]

        # Current player (0 or 1)
        self.current_player = 0

        # Game done flag
        self.done = False

        # Assign random treasure locations
        choices = np.arange(25)
        self.treasure_locations[0] = int(self.np_random.choice(choices))
        self.treasure_locations[1] = int(self.np_random.choice(choices))

        # Ensure treasures are placed at different locations
        while self.treasure_locations[0] == self.treasure_locations[1]:
            self.treasure_locations[1] = int(self.np_random.choice(choices))

        # Return initial observation and info
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), -10, True, False, {}

        # Parse action into action_type and coordinates
        if action < 25:
            action_type = "scan"
            cell = action  # cell index 0-24
        elif action < 50:
            action_type = "dig"
            cell = action - 25
        else:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check if coordinates are valid
        if cell < 0 or cell >= 25:
            # Invalid cell
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Action processing
        player = self.current_player
        opponent = 1 - self.current_player

        if action_type == "scan":
            if self.scans_remaining[player] <= 0:
                # No scans remaining
                self.done = True
                return self._get_observation(), -10, True, False, {}
            else:
                # Perform scan action
                # Compute Manhattan distance to opponent's treasure
                row, col = divmod(cell, 5)
                opp_treasure_cell = self.treasure_locations[opponent]
                opp_row, opp_col = divmod(opp_treasure_cell, 5)
                distance = abs(row - opp_row) + abs(col - opp_col)
                # Update player's grid with the distance
                self.player_grids[player][cell] = distance
                # Decrease scans_remaining
                self.scans_remaining[player] -= 1
                # Reward is -10 per valid move
                reward = -10
                self.done = False  # Game continues
        elif action_type == "dig":
            # Perform dig action
            if self.player_grids[player][cell] == -2:
                # Already dug here
                self.done = True
                return self._get_observation(), -10, True, False, {}
            else:
                if cell == self.treasure_locations[opponent]:
                    # Found the opponent's treasure!
                    self.player_grids[player][cell] = -3  # Indicate treasure found
                    reward = 1
                    self.done = True
                    # Game over, current player wins
                else:
                    # No treasure here
                    self.player_grids[player][cell] = -2  # Indicate dug, no treasure
                    reward = -10
                    self.done = False  # Game continues
        else:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Switch player if game continues
        if not self.done:
            self.current_player = 1 - self.current_player

        # Return observation, reward, done, info
        return (
            self._get_observation(),
            reward,
            self.done,
            False,
            {},
        )  # observation, reward, terminated, truncated, info

    def render(self):
        # Return a string representation of the game state from current player's perspective
        player = self.current_player
        grid = self.player_grids[player]
        scans_remaining = self.scans_remaining[player]
        # Build grid display
        grid_display = ""
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                cell_value = grid[idx]
                if cell_value == -1:
                    cell_str = " ? "
                elif cell_value == -2:
                    cell_str = " X "
                elif cell_value == -3:
                    cell_str = " T "
                else:
                    # Scanned cell with distance
                    cell_str = f" {cell_value} "
                grid_display += cell_str
            grid_display += "\n"
        s = f"Player {player+1}'s turn.\nScans remaining: {scans_remaining}\nGrid:\n{grid_display}"
        return s

    def valid_moves(self):
        # Return list of valid action indices for current player
        valid_actions = []
        player = self.current_player
        scans_remaining = self.scans_remaining[player]
        grid = self.player_grids[player]

        # Scan actions
        if scans_remaining > 0:
            for cell in range(25):
                # Allow scanning any cell
                valid_actions.append(cell)  # Actions 0-24 are scans

        # Dig actions
        for cell in range(25):
            if grid[cell] != -2 and grid[cell] != -3:
                # Can dig at cells not already dug
                valid_actions.append(cell + 25)  # Actions 25-49 are digs

        return valid_actions

    def _get_observation(self):
        # Returns the observation for the current player
        # Observation is an array of length 26
        # First 25 are grid cells, last is scans_remaining
        player = self.current_player
        grid = self.player_grids[player]
        scans_remaining = self.scans_remaining[player]
        observation = np.append(grid, scans_remaining).astype(np.int32)
        return observation
