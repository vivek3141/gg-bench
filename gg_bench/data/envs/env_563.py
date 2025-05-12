import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)

        # Define observation space: shape (5,5,4)
        # Channels:
        # 0 - current player's position
        # 1 - other player's position
        # 2 - current player's traps
        # 3 - other player's traps
        self.grid_size = 5
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.grid_size, self.grid_size, 4), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.current_player = 1  # Player 1 starts
        # Initialize player positions (both start at (0,0))
        self.positions = {1: [0, 0], 2: [0, 0]}

        # Randomly place traps for both players
        # Exclude starting position (0,0) and ending position (4,4)
        all_positions = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) != (0, 0) and (i, j) != (self.grid_size - 1, self.grid_size - 1)
        ]
        np.random.shuffle(all_positions)
        self.traps = {
            1: set(all_positions[:3]),
            2: set(all_positions[3:6]),
        }

        return self._get_observation(), {}  # Return observation and info

    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)
        # Current player's position
        cp_pos = self.positions[self.current_player]
        obs[cp_pos[0], cp_pos[1], 0] = 1.0
        # Other player's position
        other_player = 1 if self.current_player == 2 else 2
        op_pos = self.positions[other_player]
        obs[op_pos[0], op_pos[1], 1] = 1.0
        # Current player's traps
        for trap in self.traps[self.current_player]:
            obs[trap[0], trap[1], 2] = 1.0
        # Other player's traps
        for trap in self.traps[other_player]:
            obs[trap[0], trap[1], 3] = 1.0
        return obs

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map action to movement
        move_mapping = {
            0: (-1, 0),  # up
            1: (1, 0),  # down
            2: (0, -1),  # left
            3: (0, 1),  # right
        }
        if action not in move_mapping:
            # Invalid action
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        move = move_mapping[action]

        # Get current position
        cp_pos = self.positions[self.current_player]
        new_pos = [cp_pos[0] + move[0], cp_pos[1] + move[1]]

        # Check if move is within grid boundaries
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            # Invalid move
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Update position
        self.positions[self.current_player] = new_pos

        # Check for opponent's trap
        other_player = 1 if self.current_player == 2 else 2
        if tuple(new_pos) in self.traps[other_player]:
            # Player steps on opponent's trap, return to start
            self.positions[self.current_player] = [0, 0]

        # Check for win
        if new_pos == [self.grid_size - 1, self.grid_size - 1]:
            # Player has reached the goal
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Valid move, reward is -10
        reward = -10

        # Switch to other player
        self.current_player = other_player

        return self._get_observation(), reward, False, False, {}

    def render(self):
        grid = [["    " for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Mark traps
        for player_id, traps in self.traps.items():
            for trap in traps:
                grid[trap[0]][trap[1]] = f"T{player_id} "

        # Mark players
        for player_id, pos in self.positions.items():
            marker = f"P{player_id} "
            if "T" in grid[pos[0]][pos[1]]:
                grid[pos[0]][pos[1]] = grid[pos[0]][pos[1]].replace(" ", marker)
            else:
                grid[pos[0]][pos[1]] = marker

        # Build string representation
        grid_str = ""
        for row in grid:
            row_str = "|".join(cell.center(4) for cell in row)
            grid_str += f"|{row_str}|\n"
        return grid_str

    def valid_moves(self):
        cp_pos = self.positions[self.current_player]
        valid_actions = []
        if cp_pos[0] > 0:
            valid_actions.append(0)  # up
        if cp_pos[0] < self.grid_size - 1:
            valid_actions.append(1)  # down
        if cp_pos[1] > 0:
            valid_actions.append(2)  # left
        if cp_pos[1] < self.grid_size - 1:
            valid_actions.append(3)  # right
        return valid_actions
