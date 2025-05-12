import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Grid size
        self.grid_size = 5

        # Define action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)

        # Define observation space (grid with 4 channels)
        # Channels:
        # 0 - Cell content:
        #     0: empty
        #     1: Player 1
        #     2: Player 2
        #     10-14: Energy Cells with values 1 to 5 (10 + value)
        # 1 - Visited by Player 1: 1 or 0
        # 2 - Visited by Player 2: 1 or 0
        # 3 - Current player's energy score (broadcasted to the grid)
        self.observation_space = spaces.Box(
            low=0, high=20, shape=(self.grid_size, self.grid_size, 4), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Place Player 1 at (0,0), Player 2 at (4,4)
        self.p1_pos = [0, 0]
        self.p2_pos = [self.grid_size - 1, self.grid_size - 1]
        self.grid[tuple(self.p1_pos)] = 1  # Player 1
        self.grid[tuple(self.p2_pos)] = 2  # Player 2

        # Initialize energy cells
        self.energy_cells = {}

        # Randomly place 8 energy cells on the grid
        empty_cells = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in [tuple(self.p1_pos), tuple(self.p2_pos)]
        ]
        np.random.shuffle(empty_cells)
        num_energy_cells = 8
        for _ in range(num_energy_cells):
            if empty_cells:
                pos = empty_cells.pop()
                value = np.random.randint(1, 6)  # Energy value between 1 and 5
                self.grid[pos] = 10 + value  # Energy cells coded as 10 + value
                self.energy_cells[pos] = value

        # Initialize visited cells
        self.visited_p1 = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.visited_p2 = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Mark starting positions as visited
        self.visited_p1[tuple(self.p1_pos)] = 1
        self.visited_p2[tuple(self.p2_pos)] = 1

        # Initialize scores
        self.p1_score = 0
        self.p2_score = 0

        # Current player: 1 or 2
        self.current_player = 1

        # Game over flag
        self.done = False

        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Map action to movement
        # 0: up (-1, 0)
        # 1: down (+1, 0)
        # 2: left (0, -1)
        # 3: right (0, +1)
        move_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        if action not in [0, 1, 2, 3]:
            # Invalid action
            return self._get_obs(), -10, True, False, {}

        dx, dy = move_map[action]

        if self.current_player == 1:
            pos = self.p1_pos
            opponent_pos = self.p2_pos
            visited = self.visited_p1
        else:
            pos = self.p2_pos
            opponent_pos = self.p1_pos
            visited = self.visited_p2

        new_x = pos[0] + dx
        new_y = pos[1] + dy

        # Check bounds
        if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
            return self._get_obs(), -10, True, False, {}

        new_pos = [new_x, new_y]

        # Check if moving into opponent's cell
        if new_pos == opponent_pos:
            return self._get_obs(), -10, True, False, {}

        # Check if revisiting a cell
        if visited[new_x, new_y] == 1:
            return self._get_obs(), -10, True, False, {}

        # Move is valid

        # Update visited cells
        visited[new_x, new_y] = 1

        # Update grid
        # Remove player's marker from current position
        self.grid[tuple(pos)] = 0
        # Place player's marker at new position
        self.grid[new_x, new_y] = self.current_player

        # Update player's position
        if self.current_player == 1:
            self.p1_pos = new_pos
        else:
            self.p2_pos = new_pos

        # Check for capturing an Energy Cell
        reward = 0
        terminated = False

        if (new_x, new_y) in self.energy_cells:
            value = self.energy_cells.pop((new_x, new_y))

            # Increase player's energy score
            if self.current_player == 1:
                self.p1_score += value
            else:
                self.p2_score += value

            # Energy Cell is removed from grid (player marker is already set)
            pass

        # Check for victory
        if self.current_player == 1:
            if self.p1_score >= 15:
                reward = 1
                terminated = True
                self.done = True
        else:
            if self.p2_score >= 15:
                reward = 1
                terminated = True
                self.done = True

        # Switch current player if game not over
        if not self.done:
            self.current_player = 2 if self.current_player == 1 else 1

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        grid_repr = ""
        for i in range(self.grid_size):
            grid_repr += "["
            for j in range(self.grid_size):
                cell = self.grid[i, j]
                if cell == 0:
                    grid_repr += "    "
                elif cell == 1:
                    grid_repr += " P1 "
                elif cell == 2:
                    grid_repr += " P2 "
                elif cell >= 10:
                    grid_repr += f"E{cell - 10} "
                else:
                    grid_repr += "????"
                if j != self.grid_size - 1:
                    grid_repr += "|"
            grid_repr += "]\n"
        return grid_repr

    def valid_moves(self):
        # Return list of valid action indices for the current player
        valid_actions = []
        move_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        if self.current_player == 1:
            pos = self.p1_pos
            opponent_pos = self.p2_pos
            visited = self.visited_p1
        else:
            pos = self.p2_pos
            opponent_pos = self.p1_pos
            visited = self.visited_p2

        for action, (dx, dy) in move_map.items():
            new_x = pos[0] + dx
            new_y = pos[1] + dy

            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                new_pos = [new_x, new_y]
                if new_pos != opponent_pos and visited[new_x, new_y] == 0:
                    valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        # Construct the observation
        observation = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.int32)

        # Channel 0: Cell content
        observation[:, :, 0] = self.grid

        # Channel 1: Visited by Player 1
        observation[:, :, 1] = self.visited_p1

        # Channel 2: Visited by Player 2
        observation[:, :, 2] = self.visited_p2

        # Channel 3: Current player's energy score (broadcasted)
        if self.current_player == 1:
            observation[:, :, 3] = self.p1_score
        else:
            observation[:, :, 3] = self.p2_score

        return observation
