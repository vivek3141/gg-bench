import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space is Discrete(81)
        # Each action corresponds to (unit_number, cell_index)
        self.action_space = spaces.Discrete(81)

        # Observation space is a Box space, shape (18,), values between -1 and 1
        # First 9 elements: Grid representation
        # Next 9 elements: Available units for current player
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.units_placed = [None] * 9  # list of (player, unit_value) or None
        self.current_player = 1  # 1 or -1
        self.available_units = {
            1: set(range(1, 10)),  # Player 1's available units
            -1: set(range(1, 10)),  # Player 2's available units
        }
        self.done = False

        # Return initial observation and empty info
        return self._get_observation(), {}

    def _get_observation(self):
        # Observation is a numpy array of shape (18,)
        obs = np.zeros(18, dtype=np.float32)
        # First 9 elements represent the grid
        for i in range(9):
            if self.units_placed[i] is None:
                obs[i] = 0.0
            else:
                player, unit_value = self.units_placed[i]
                value = unit_value / 9.0  # normalize between 1/9 to 1
                if player == self.current_player:
                    obs[i] = value  # positive value
                else:
                    obs[i] = -value  # negative value
        # Next 9 elements represent available units for current player
        for unit in range(1, 10):
            if unit in self.available_units[self.current_player]:
                obs[9 + unit - 1] = 1.0  # available
            else:
                obs[9 + unit - 1] = 0.0  # used
        return obs

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action to (unit_number, cell_index)
        unit_idx = action // 9  # 0..8
        cell_idx = action % 9  # 0..8
        unit_number = unit_idx + 1  # units are from 1 to 9

        # Validate move
        if unit_number not in self.available_units[self.current_player]:
            # Invalid move: unit not available
            self.done = True
            return self._get_observation(), -10, True, False, {}
        if self.units_placed[cell_idx] is not None:
            # Invalid move: cell already occupied
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move
        self.units_placed[cell_idx] = (self.current_player, unit_number)
        self.available_units[self.current_player].remove(unit_number)

        # Check if game is over
        if all(cell is not None for cell in self.units_placed):
            self.done = True
            # Calculate scores and determine winner
            player_scores = self._calculate_scores()
            if player_scores[self.current_player] > player_scores[-self.current_player]:
                # Current player wins
                reward = 1
            else:
                reward = 0
            return self._get_observation(), reward, True, False, {}
        else:
            # Game not over, switch player
            self.current_player *= -1
            return self._get_observation(), 0, False, False, {}

    def _calculate_scores(self):
        player_scores = {1: 0, -1: 0}
        # For each cell
        for idx in range(9):
            if self.units_placed[idx] is not None:
                player, unit_value = self.units_placed[idx]
                # Base power is unit_value
                base_power = unit_value
                adjacency_bonus = 0
                adjacent_indices = self._get_adjacent_indices(idx)
                for adj_idx in adjacent_indices:
                    adj_cell = self.units_placed[adj_idx]
                    if adj_cell is not None:
                        adj_player, adj_unit_value = adj_cell
                        if adj_player == player:
                            adjacency_bonus += adj_unit_value
                unit_power = base_power + adjacency_bonus
                player_scores[player] += unit_power
        return player_scores

    def _get_adjacent_indices(self, idx):
        # idx is from 0 to 8
        row = idx // 3
        col = idx % 3
        adjacent_indices = []
        if row > 0:
            adjacent_indices.append((row - 1) * 3 + col)
        if row < 2:
            adjacent_indices.append((row + 1) * 3 + col)
        if col > 0:
            adjacent_indices.append(row * 3 + (col - 1))
        if col < 2:
            adjacent_indices.append(row * 3 + (col + 1))
        return adjacent_indices

    def render(self):
        board_str = "-------------\n"
        for row in range(3):
            board_str += "|"
            for col in range(3):
                idx = row * 3 + col
                if self.units_placed[idx] is None:
                    board_str += "     |"
                else:
                    player, unit_value = self.units_placed[idx]
                    symbol = "P1" if player == 1 else "P2"
                    board_str += f"{symbol}:{unit_value}|"
            board_str += "\n-------------\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        available_units = self.available_units[self.current_player]
        empty_cells = [idx for idx in range(9) if self.units_placed[idx] is None]
        for unit_number in available_units:
            unit_idx = unit_number - 1  # units from 1..9 correspond to indices 0..8
            for cell_idx in empty_cells:
                action = unit_idx * 9 + cell_idx
                valid_actions.append(action)
        return valid_actions
