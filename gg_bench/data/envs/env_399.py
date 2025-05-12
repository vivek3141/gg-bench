import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Grid dimensions
        self.grid_rows = 5
        self.grid_cols = 5
        self.num_tiles = self.grid_rows * self.grid_cols  # Total of 25 tiles

        # Action Mapping:
        # Actions 0-24: Flip tile at index 0-24
        # Actions 25 onward: Quantum Flip actions on adjacent tile pairs
        self.tile_indices = list(range(self.num_tiles))

        # Precompute adjacent tile pairs for Quantum Flip actions
        self.adjacent_pairs = []
        for index in range(self.num_tiles):
            row = index // self.grid_cols
            col = index % self.grid_cols

            neighbors = []
            # Up
            if row > 0:
                neighbors.append((row - 1) * self.grid_cols + col)
            # Down
            if row < self.grid_rows - 1:
                neighbors.append((row + 1) * self.grid_cols + col)
            # Left
            if col > 0:
                neighbors.append(row * self.grid_cols + (col - 1))
            # Right
            if col < self.grid_cols - 1:
                neighbors.append(row * self.grid_cols + (col + 1))

            for neighbor in neighbors:
                if index < neighbor:
                    self.adjacent_pairs.append((index, neighbor))

        self.num_actions = self.num_tiles + len(self.adjacent_pairs)
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space: tiles (25), scores (2), Quantum Flip used flags (2)
        obs_low = np.array([-5] * self.num_tiles + [0, 0, 0, 0], dtype=np.int32)
        obs_high = np.array([5] * self.num_tiles + [21, 21, 1, 1], dtype=np.int32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.int32)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly assign tile values between 1 and 5
        self.tile_values = self.np_random.randint(1, 6, size=self.num_tiles)
        self.tile_states = np.zeros(self.num_tiles, dtype=np.int32)  # 0: unflipped
        self.current_player = 1  # Player 1 starts; Player 2 is represented by -1
        self.scores = {1: 0, -1: 0}
        self.quantum_flip_used = {1: False, -1: False}
        self.done = False

        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        reward = 0

        if action < self.num_tiles:
            # Flip a single tile
            tile_index = action
            self._flip_tile(tile_index)
        else:
            # Quantum Flip action
            qf_index = action - self.num_tiles
            self._perform_quantum_flip(qf_index)

        # Check win/lose conditions
        if self.scores[self.current_player] == 21:
            self.done = True
            reward = 1
        elif self.scores[self.current_player] > 21:
            self.done = True
            reward = -1
        else:
            # Switch to the other player
            self.current_player *= -1

        return self._get_observation(), reward, self.done, False, {}

    def _flip_tile(self, tile_index):
        self.tile_states[tile_index] = self.current_player
        tile_value = self.tile_values[tile_index]
        self.scores[self.current_player] += tile_value

    def _perform_quantum_flip(self, qf_index):
        if self.quantum_flip_used[self.current_player]:
            self.done = True
            return

        tile1, tile2 = self.adjacent_pairs[qf_index]
        if self.tile_states[tile1] != 0 or self.tile_states[tile2] != 0:
            self.done = True
            return

        value1 = self.tile_values[tile1]
        value2 = self.tile_values[tile2]

        # Choose the value that keeps the score less than or equal to 21
        potential_scores = {
            tile1: self.scores[self.current_player] + value1,
            tile2: self.scores[self.current_player] + value2,
        }

        # Filter out scores that exceed 21
        valid_choices = {
            tile: score for tile, score in potential_scores.items() if score <= 21
        }

        if valid_choices:
            # Choose the tile leading to the highest valid score
            chosen_tile = max(valid_choices, key=valid_choices.get)
        else:
            # Choose the tile that minimizes the overshoot
            chosen_tile = min(potential_scores, key=potential_scores.get)

        self.tile_states[chosen_tile] = self.current_player
        chosen_value = self.tile_values[chosen_tile]
        self.scores[self.current_player] += chosen_value
        # The unchosen tile remains unflipped

        self.quantum_flip_used[self.current_player] = True

    def render(self):
        grid_repr = ""
        for row in range(self.grid_rows):
            row_repr = ""
            for col in range(self.grid_cols):
                index = row * self.grid_cols + col
                state = self.tile_states[index]
                if state == 0:
                    row_repr += "[ ] "
                else:
                    value = self.tile_values[index]
                    row_repr += f"[{value}] "
            grid_repr += row_repr + "\n"
        grid_repr += f"Player 1 Score: {self.scores[1]}\n"
        grid_repr += f"Player 2 Score: {self.scores[-1]}\n"
        return grid_repr

    def valid_moves(self):
        if self.done:
            return []

        moves = []
        # Add valid tile flip actions
        for index in range(self.num_tiles):
            if self.tile_states[index] == 0:
                moves.append(index)
        # Add valid Quantum Flip actions
        if not self.quantum_flip_used[self.current_player]:
            for qf_index, (tile1, tile2) in enumerate(self.adjacent_pairs):
                if self.tile_states[tile1] == 0 and self.tile_states[tile2] == 0:
                    moves.append(self.num_tiles + qf_index)
        return moves

    def _get_observation(self):
        tile_obs = np.zeros(self.num_tiles, dtype=np.int32)
        for index in range(self.num_tiles):
            state = self.tile_states[index]
            if state == 0:
                tile_obs[index] = 0
            elif state == self.current_player:
                tile_obs[index] = self.tile_values[index]
            else:
                tile_obs[index] = -self.tile_values[index]

        current_player_score = self.scores[self.current_player]
        opponent_player_score = self.scores[-self.current_player]
        current_qf_used = int(self.quantum_flip_used[self.current_player])
        opponent_qf_used = int(self.quantum_flip_used[-self.current_player])

        observation = np.concatenate(
            [
                tile_obs,
                [
                    current_player_score,
                    opponent_player_score,
                    current_qf_used,
                    opponent_qf_used,
                ],
            ]
        )
        return observation
