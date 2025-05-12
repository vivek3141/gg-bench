import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - up, 1 - down, 2 - left, 3 - right
        self.action_space = spaces.Discrete(4)

        # Define observation space: a 5x5 grid with 3 layers (cell type, bases, spies)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(5, 5, 3), dtype=np.int8
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid: 0 - unknown, 1 - safe cell, 2 - trap cell
        self.grid = np.zeros((5, 5), dtype=np.int8)

        # Randomly place 5 traps, excluding the base positions
        all_positions = [(i, j) for i in range(5) for j in range(5)]
        all_positions.remove((0, 0))  # Player 1's base
        all_positions.remove((4, 4))  # Player 2's base
        self.trap_positions = random.sample(all_positions, 5)

        # Initialize player positions
        self.player_positions = {
            1: (0, 0),  # Player 1's starting position
            2: (4, 4),  # Player 2's starting position
        }

        # Set up bases layer: 0 - no base, 1 - Player 1's base, 2 - Player 2's base
        self.bases = np.zeros((5, 5), dtype=np.int8)
        self.bases[0, 0] = 1  # Player 1's base
        self.bases[4, 4] = 2  # Player 2's base

        # Set up spies layer: 0 - no spy, 1 - Player 1's spy, 2 - Player 2's spy
        self.spies = np.zeros((5, 5), dtype=np.int8)
        self.spies[0, 0] = 1  # Player 1's spy
        self.spies[4, 4] = 2  # Player 2's spy

        # Current player (1 or 2)
        self.current_player = 1

        # Game over flag
        self.done = False

        # Return initial observation and info
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action to movement
        action_map = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        if action not in action_map:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        dx, dy = action_map[action]
        x, y = self.player_positions[self.current_player]
        new_x, new_y = x + dx, y + dy

        # Check grid boundaries
        if not (0 <= new_x < 5 and 0 <= new_y < 5):
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Move the spy
        self.spies[x, y] = 0  # Remove spy from old position
        self.player_positions[self.current_player] = (new_x, new_y)
        self.spies[new_x, new_y] = self.current_player  # Place spy at new position

        # Reveal the cell
        if (new_x, new_y) in self.trap_positions:
            # Hit a trap
            self.grid[new_x, new_y] = 2  # Mark trap cell
            # Send spy back to base
            base_x, base_y = (0, 0) if self.current_player == 1 else (4, 4)
            self.spies[new_x, new_y] = 0  # Remove spy from trap cell
            self.spies[base_x, base_y] = self.current_player  # Spy back to base
            self.player_positions[self.current_player] = (base_x, base_y)
            # Switch to the next player
            self.current_player = 2 if self.current_player == 1 else 1
            return self._get_observation(), 0, False, False, {}
        else:
            # Safe cell
            self.grid[new_x, new_y] = 1  # Mark safe cell

        # Check for victory condition
        opponent_base = (0, 0) if self.current_player == 2 else (4, 4)
        if (new_x, new_y) == opponent_base:
            # Current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1
        return self._get_observation(), 0, False, False, {}

    def render(self):
        render_str = ""
        for i in range(5):
            row_str = ""
            for j in range(5):
                cell_str = ""
                # Check for spy
                if self.spies[i, j] == 1:
                    cell_str = "S1"
                elif self.spies[i, j] == 2:
                    cell_str = "S2"
                # Check for base
                elif self.bases[i, j] == 1:
                    cell_str = "B1"
                elif self.bases[i, j] == 2:
                    cell_str = "B2"
                else:
                    # Cell type
                    if self.grid[i, j] == 0:
                        cell_str = " ? "
                    elif self.grid[i, j] == 1:
                        cell_str = " . "
                    elif self.grid[i, j] == 2:
                        cell_str = " X "
                if len(cell_str) == 2:
                    cell_str = " " + cell_str + " "
                row_str += "[" + cell_str + "]"
            render_str += row_str + "\n"
        return render_str

    def valid_moves(self):
        x, y = self.player_positions[self.current_player]
        moves = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }
        valid_actions = []
        for action, (dx, dy) in moves.items():
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 5 and 0 <= new_y < 5:
                valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Combine grid layers into observation
        obs_grid = np.zeros((5, 5, 3), dtype=np.int8)
        obs_grid[:, :, 0] = self.grid  # Cell types
        obs_grid[:, :, 1] = self.bases  # Bases
        obs_grid[:, :, 2] = self.spies  # Spies positions
        return obs_grid
