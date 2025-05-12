import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        # 0: Move Up
        # 1: Move Down
        # 2: Move Left
        # 3: Move Right
        # 4-28: Place Trap at cell indices 0-24 (grid cells flattened)
        self.action_space = spaces.Discrete(29)

        # Define observation space
        # 0: Empty
        # 1: Player A
        # 2: Player B
        # 3: Treasure
        # 4: Trap
        self.observation_space = spaces.Box(low=0, high=4, shape=(5, 5), dtype=np.int8)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)

        # Set initial positions
        self.grid[0, 0] = 1  # Player A
        self.grid[4, 4] = 2  # Player B
        self.grid[2, 2] = 3  # Treasure

        # Player states
        self.player_positions = {
            1: [0, 0],  # Player A position
            2: [4, 4],  # Player B position
        }
        self.player_traps = {
            1: 3,  # Player A traps remaining
            2: 3,  # Player B traps remaining
        }
        self.player_has_treasure = {
            1: False,
            2: False,
        }

        # Game variables
        self.current_player = 1  # Player A starts
        self.opponent_player = 2
        self.done = False
        self.trapped_players = {
            1: False,
            2: False,
        }
        self.skip_turn_players = {
            1: False,
            2: False,
        }
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        reward = -10  # Default reward for a valid move
        info = {}

        # Check if current player needs to skip turn
        if self.skip_turn_players[self.current_player]:
            self.skip_turn_players[self.current_player] = False  # Skip only one turn
            self._switch_player()
            return self._get_obs(), reward, self.done, False, info

        # Validate action
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            reward = -15  # Penalty for invalid action
            self.done = True
            return self._get_obs(), reward, True, False, info

        # Handle action
        if action in [0, 1, 2, 3]:  # Movement actions
            moved = self._move_player(action)
            if not moved:
                reward = -15  # Penalty for invalid movement
                self.done = True
                return self._get_obs(), reward, True, False, info
        else:  # Trap placement actions
            placed = self._place_trap(action)
            if not placed:
                reward = -15  # Penalty for invalid trap placement
                self.done = True
                return self._get_obs(), reward, True, False, info

        # Check for win condition
        if self.player_has_treasure[self.current_player]:
            if self.player_positions[self.current_player] == self._starting_position(
                self.current_player
            ):
                reward = 1  # Current player wins
                self.done = True
                return self._get_obs(), reward, True, False, info

        # Switch to next player
        self._switch_player()
        return self._get_obs(), reward, self.done, False, info

    def render(self):
        grid_repr = ""
        for y in range(5):
            row = ""
            for x in range(5):
                cell = self.grid[y, x]
                if cell == 0:
                    row += "[ ]"
                elif cell == 1:
                    row += "[A]"
                elif cell == 2:
                    row += "[B]"
                elif cell == 3:
                    row += "[T]"
                elif cell == 4:
                    row += "[X]"
            grid_repr += row + "\n"
        return grid_repr

    def valid_moves(self):
        actions = []
        # If player is trapped and needs to skip turn
        if self.skip_turn_players[self.current_player]:
            return actions

        # Movement actions
        y, x = self.player_positions[self.current_player]
        if not self.trapped_players[self.current_player]:
            directions = {
                0: (-1, 0),  # Up
                1: (1, 0),  # Down
                2: (0, -1),  # Left
                3: (0, 1),  # Right
            }
            for dir_action, (dy, dx) in directions.items():
                ny, nx = y + dy, x + dx
                if 0 <= ny < 5 and 0 <= nx < 5:
                    target_cell = self.grid[ny, nx]
                    if target_cell not in [1, 2, 4]:  # Can't move onto player or trap
                        actions.append(dir_action)

        # Trap placement actions
        if self.player_traps[self.current_player] > 0:
            for idx in range(25):
                ty, tx = divmod(idx, 5)
                cell = self.grid[ty, tx]
                if cell == 0:
                    actions.append(4 + idx)
                elif cell == 3:
                    # Treasure is considered an empty cell after being picked up
                    if not any(self.player_has_treasure.values()):
                        continue
                    else:
                        actions.append(4 + idx)
        return actions

    def _get_obs(self):
        return self.grid.copy()

    def _move_player(self, action):
        y, x = self.player_positions[self.current_player]

        directions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }
        dy, dx = directions[action]
        ny, nx = y + dy, x + dx

        # Check if new position is within bounds
        if not (0 <= ny < 5 and 0 <= nx < 5):
            return False

        target_cell = self.grid[ny, nx]

        # Can't move onto opponent or trap
        if target_cell in [1, 2, 4]:
            return False

        # Move player
        self.grid[y, x] = 0  # Clear old position
        self.grid[ny, nx] = self.current_player  # Set new position
        self.player_positions[self.current_player] = [ny, nx]

        # Check for Treasure
        if target_cell == 3:
            self.player_has_treasure[self.current_player] = True
            self.grid[ny, nx] = self.current_player  # Player now carrying treasure

        # Check for Trap
        if target_cell == 4:
            self.skip_turn_players[self.current_player] = True  # Skip next turn

        return True

    def _place_trap(self, action):
        trap_idx = action - 4
        y, x = divmod(trap_idx, 5)

        # Check if trap can be placed at the location
        cell = self.grid[y, x]
        if cell != 0:
            return False

        # Place trap
        self.grid[y, x] = 4  # Set trap
        self.player_traps[self.current_player] -= 1
        return True

    def _switch_player(self):
        self.current_player, self.opponent_player = (
            self.opponent_player,
            self.current_player,
        )

    def _starting_position(self, player):
        if player == 1:
            return [0, 0]
        else:
            return [4, 4]
