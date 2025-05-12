import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0-3: Move up, down, left, right
        #          4-28: Place wall at cell index 0-24
        self.action_space = spaces.Discrete(29)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Place player tokens on random empty cells
        empty_cells = [(i, j) for i in range(5) for j in range(5)]
        self.np_random.shuffle(empty_cells)
        p1_pos = empty_cells.pop()
        p2_pos = empty_cells.pop()
        self.grid[p1_pos] = self.current_player  # Player 1 token
        self.grid[p2_pos] = -self.current_player  # Player 2 token

        # Store players' positions
        self.player_positions = {
            self.current_player: p1_pos,
            -self.current_player: p2_pos,
        }

        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        reward = 0

        # Get opponent's player number
        opponent = -self.current_player

        # Determine action type
        if action in [0, 1, 2, 3]:  # Move action
            direction = action
            # Get current player position
            x, y = self.player_positions[self.current_player]
            if direction == 0:  # Up
                new_x, new_y = x - 1, y
            elif direction == 1:  # Down
                new_x, new_y = x + 1, y
            elif direction == 2:  # Left
                new_x, new_y = x, y - 1
            elif direction == 3:  # Right
                new_x, new_y = x, y + 1

            # Check if move is valid
            if not self._is_within_bounds(new_x, new_y) or self.grid[new_x, new_y] != 0:
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, {}
            else:
                # Move the token
                self.grid[x, y] = 0
                self.grid[new_x, new_y] = self.current_player
                self.player_positions[self.current_player] = (new_x, new_y)
        elif action in range(4, 29):  # Place wall action
            cell_index = action - 4
            x = cell_index // 5
            y = cell_index % 5
            # Check if placement is valid
            if self.grid[x, y] != 0:
                self.done = True
                reward = -10
                return self._get_observation(), reward, True, False, {}
            else:
                # Place the wall
                self.grid[x, y] = 2  # Wall
        else:
            # Invalid action
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Check if opponent can move
        if not self._opponent_can_move(opponent):
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}

        # Switch player
        self.current_player *= -1

        return self._get_observation(), reward, False, False, {}

    def render(self):
        grid_str = "  1 2 3 4 5\n"
        for i in range(5):
            row_str = f"{i+1} "
            for j in range(5):
                cell = self.grid[i, j]
                if cell == 0:
                    row_str += ". "
                elif cell == 1:
                    row_str += "X "
                elif cell == -1:
                    row_str += "O "
                elif cell == 2:
                    row_str += "# "
            grid_str += row_str + "\n"
        grid_str += f"Current player: {'X' if self.current_player == 1 else 'O'}\n"
        return grid_str

    def valid_moves(self):
        moves = []
        # Move actions
        x, y = self.player_positions[self.current_player]
        directions = [0, 1, 2, 3]
        for dir in directions:
            if dir == 0:  # Up
                new_x, new_y = x - 1, y
            elif dir == 1:  # Down
                new_x, new_y = x + 1, y
            elif dir == 2:  # Left
                new_x, new_y = x, y - 1
            elif dir == 3:  # Right
                new_x, new_y = x, y + 1
            if self._is_within_bounds(new_x, new_y) and self.grid[new_x, new_y] == 0:
                moves.append(dir)
        # Wall placement actions
        for idx in range(25):
            x = idx // 5
            y = idx % 5
            if self.grid[x, y] == 0:
                moves.append(idx + 4)
        return moves

    def _opponent_can_move(self, opponent):
        x, y = self.player_positions[opponent]
        directions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for new_x, new_y in directions:
            if self._is_within_bounds(new_x, new_y) and self.grid[new_x, new_y] == 0:
                return True
        return False

    def _get_observation(self):
        # Return the grid with current player's perspective
        # Replace player's token with 1, opponent's with -1
        obs_grid = self.grid.copy()
        obs_grid[obs_grid == self.current_player] = 1
        obs_grid[obs_grid == -self.current_player] = -1
        return obs_grid

    def _is_within_bounds(self, x, y):
        return 0 <= x < 5 and 0 <= y < 5
