import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 8 possible moves for both Mouse and Cat
        self.action_space = spaces.Discrete(8)

        # Define observation space: Flattened 5x5 grid, values 0 (empty), 1 (Mouse), 2 (Cat)
        self.observation_space = spaces.Box(low=0, high=2, shape=(25,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the 5x5 grid
        self.grid = np.zeros((5, 5), dtype=np.int32)

        # Set starting positions
        self.mouse_pos = [0, 0]  # (x, y)
        self.cat_pos = [0, 4]  # (x, y)
        self.cheese_pos = [4, 4]  # (x, y), always same

        # Place Mouse and Cat on the grid
        self.grid[self.mouse_pos[1], self.mouse_pos[0]] = 1  # Mouse
        self.grid[self.cat_pos[1], self.cat_pos[0]] = 2  # Cat
        # Cheese is not placed on the grid as it's always at (4,4)

        self.current_player = "mouse"  # 'mouse' or 'cat'
        self.done = False

        observation = self._get_observation()
        return observation, {}  # observation, info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}  # Game is over

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Perform the action
        if self.current_player == "mouse":
            dx, dy = self._mouse_action_to_delta(action)
            new_pos = [self.mouse_pos[0] + dx, self.mouse_pos[1] + dy]
            self.grid[self.mouse_pos[1], self.mouse_pos[0]] = 0  # Clear old position
            self.mouse_pos = new_pos
            self.grid[self.mouse_pos[1], self.mouse_pos[0]] = 1  # Set new position

            # Check for win condition
            if self.mouse_pos == self.cheese_pos:
                self.done = True
                return self._get_observation(), 1, True, False, {}
            if self.mouse_pos == self.cat_pos:
                # Cat catches the mouse
                self.done = True
                return self._get_observation(), -1, True, False, {}

            self.current_player = "cat"

        elif self.current_player == "cat":
            dx, dy = self._cat_action_to_delta(action)
            path_clear = self._check_cat_path(dx, dy)
            if not path_clear:
                # Invalid move (path blocked or moves off the grid)
                self.done = True
                return self._get_observation(), -10, True, False, {}

            new_pos = [self.cat_pos[0] + dx, self.cat_pos[1] + dy]
            self.grid[self.cat_pos[1], self.cat_pos[0]] = 0  # Clear old position
            self.cat_pos = new_pos
            self.grid[self.cat_pos[1], self.cat_pos[0]] = 2  # Set new position

            # Check for win condition
            if self.cat_pos == self.mouse_pos:
                self.done = True
                return self._get_observation(), 1, True, False, {}

            self.current_player = "mouse"

        # No one has won yet
        return self._get_observation(), 0, False, False, {}

    def render(self):
        grid_str = ""
        for y in range(5):
            for x in range(5):
                if [x, y] == self.mouse_pos:
                    grid_str += "M "
                elif [x, y] == self.cat_pos:
                    grid_str += "C "
                elif [x, y] == self.cheese_pos:
                    grid_str += "X "
                else:
                    grid_str += ". "
            grid_str += "\n"
        return grid_str

    def valid_moves(self):
        moves = []
        if self.current_player == "mouse":
            for action in range(8):
                dx, dy = self._mouse_action_to_delta(action)
                new_x = self.mouse_pos[0] + dx
                new_y = self.mouse_pos[1] + dy
                if 0 <= new_x < 5 and 0 <= new_y < 5:
                    moves.append(action)
        elif self.current_player == "cat":
            for action in range(8):
                dx, dy = self._cat_action_to_delta(action)
                new_x = self.cat_pos[0] + dx
                new_y = self.cat_pos[1] + dy
                if 0 <= new_x < 5 and 0 <= new_y < 5:
                    # Check if path is clear for moves of distance 2
                    if abs(dx) == 2 or abs(dy) == 2:
                        if self._check_cat_path(dx, dy):
                            moves.append(action)
                    else:
                        moves.append(action)
        return moves

    def _get_observation(self):
        return self.grid.flatten()

    def _mouse_action_to_delta(self, action):
        # Map actions 0-7 to movements for Mouse
        # 0:N, 1:NE, 2:E, 3:SE, 4:S, 5:SW, 6:W, 7:NW
        deltas = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        return deltas[action]

    def _cat_action_to_delta(self, action):
        # Map actions 0-7 to movements for Cat
        # 0:Up 1, 1:Up 2, 2:Down 1, 3:Down 2, 4:Left 1, 5:Left 2, 6:Right 1, 7:Right 2
        deltas = [(0, -1), (0, -2), (0, 1), (0, 2), (-1, 0), (-2, 0), (1, 0), (2, 0)]
        return deltas[action]

    def _check_cat_path(self, dx, dy):
        # Check if Cat's path is within the grid and not blocked
        steps = max(abs(dx), abs(dy))
        step_dx = int(dx / steps) if steps != 0 else 0
        step_dy = int(dy / steps) if steps != 0 else 0
        for step in range(1, steps + 1):
            new_x = self.cat_pos[0] + step_dx * step
            new_y = self.cat_pos[1] + step_dy * step
            if not (0 <= new_x < 5 and 0 <= new_y < 5):
                return False  # Moves off the grid
            # Can't change direction between the two moves; must be in straight line
        return True
