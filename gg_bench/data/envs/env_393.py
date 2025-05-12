import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to moving in a direction with a certain number of steps
        # There are 4 directions and up to 6 steps (max in a 7x7 grid)
        self.directions = ["up", "down", "left", "right"]
        self.max_steps = 6
        self.action_space = spaces.Discrete(len(self.directions) * self.max_steps)

        # Observation space is a 7x7 grid
        # 0: unvisited cell
        # 1: visited cell
        # 2: current position of the token
        self.grid_size = 7
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.grid_size, self.grid_size), dtype=np.int8
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Create a 7x7 grid initialized to 0 (unvisited)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Starting position is at the center of the grid (4,4)
        self.current_position = (3, 3)  # Zero-based indexing
        self.grid[self.current_position] = 1  # Mark the starting cell as visited

        # Current player: 1 or 2
        self.current_player = 1
        self.done = False

        # Keep track of visited cells
        self.visited = set()
        self.visited.add(self.current_position)

        # Return the observation and info
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Map the action to direction and steps
        direction_index = action % len(self.directions)
        steps = action // len(self.directions) + 1  # Steps from 1 to 6

        direction = self.directions[direction_index]

        # Try to move the token
        path_positions = self._get_path_positions(direction, steps)

        if not path_positions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}
        else:
            # Valid move
            # Mark the cells as visited
            for pos in path_positions:
                self.grid[pos] = 1
                self.visited.add(pos)

            # Update the token position
            self.current_position = path_positions[-1]

            # Check if the opponent has any valid moves
            opponent_valid_moves = self.valid_moves()
            if not opponent_valid_moves:
                # Current player wins
                self.done = True
                return self._get_observation(), 1, True, False, {}
            else:
                # Switch to the next player
                self.current_player = 2 if self.current_player == 1 else 1
                return self._get_observation(), -10, False, False, {}

    def render(self):
        # Generate a string representation of the grid
        grid_str = "  " + " ".join(str(i + 1) for i in range(self.grid_size)) + "\n"
        for i in range(self.grid_size):
            row_str = str(i + 1) + " "
            for j in range(self.grid_size):
                if (i, j) == self.current_position:
                    row_str += " T"
                elif self.grid[i, j] == 0:
                    row_str += " ."
                else:
                    row_str += " X"
            grid_str += row_str + "\n"
        return grid_str

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for steps in range(1, self.max_steps + 1):
            for direction_index, direction in enumerate(self.directions):
                action = (steps - 1) * len(self.directions) + direction_index
                path_positions = self._get_path_positions(direction, steps)
                if path_positions:
                    valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Return the current grid with the token position marked as 2
        observation = self.grid.copy()
        observation[self.current_position] = 2
        return observation

    def _get_path_positions(self, direction, steps):
        # Compute the positions the token would move through for a given action
        path_positions = []
        x, y = self.current_position

        for step in range(1, steps + 1):
            if direction == "up":
                new_x = x - step
                new_y = y
            elif direction == "down":
                new_x = x + step
                new_y = y
            elif direction == "left":
                new_x = x
                new_y = y - step
            elif direction == "right":
                new_x = x
                new_y = y + step

            # Check boundaries
            if (
                new_x < 0
                or new_x >= self.grid_size
                or new_y < 0
                or new_y >= self.grid_size
            ):
                return None  # Invalid move

            # Check if the path cells are unvisited
            path = self._get_line_positions(self.current_position, (new_x, new_y))
            for pos in path:
                if pos in self.visited:
                    return None  # Invalid move
            path_positions = path

        return path_positions

    def _get_line_positions(self, start, end):
        # Get the positions between start and end inclusive
        positions = []
        x1, y1 = start
        x2, y2 = end

        if x1 == x2:
            # Horizontal move
            step = 1 if y2 > y1 else -1
            for y in range(y1, y2 + step, step):
                positions.append((x1, y))
        elif y1 == y2:
            # Vertical move
            step = 1 if x2 > x1 else -1
            for x in range(x1, x2 + step, step):
                positions.append((x, y1))
        else:
            # Invalid move (not in a straight line)
            return []

        return positions
