import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 8 possible actions: 4 directions * 2 distances
        self.action_space = spaces.Discrete(8)
        # Observation space is the positions of both players: (x1, y1, x2, y2)
        self.observation_space = spaces.Box(low=1, high=5, shape=(4,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Initialize player positions
        self.player1_pos = np.array([1, 1], dtype=np.int32)
        self.player2_pos = np.array([5, 5], dtype=np.int32)

        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}  # Game is over

        # Get current and opponent positions
        if self.current_player == 1:
            current_pos = self.player1_pos.copy()
            opponent_pos = self.player2_pos.copy()
        else:
            current_pos = self.player2_pos.copy()
            opponent_pos = self.player1_pos.copy()

        # Decode action into direction and number of spaces
        direction, num_spaces = self._action_to_movement(action)

        # Compute new position
        new_pos = current_pos.copy()
        if direction == "up":
            new_pos[1] += num_spaces
        elif direction == "down":
            new_pos[1] -= num_spaces
        elif direction == "left":
            new_pos[0] -= num_spaces
        elif direction == "right":
            new_pos[0] += num_spaces

        # Validate move
        if not self._is_valid_move(current_pos, new_pos, opponent_pos):
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid move

        # Update position
        if self.current_player == 1:
            self.player1_pos = new_pos
        else:
            self.player2_pos = new_pos

        # Check for capture
        if np.array_equal(new_pos, opponent_pos):
            self.done = True
            return self._get_obs(), 1, True, False, {}  # Current player wins

        # Switch turn
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        return self._get_obs(), 0, False, False, {}  # Continue game

    def render(self):
        # Create a 5x5 grid
        grid = [["  " for _ in range(5)] for _ in range(5)]
        x1, y1 = self.player1_pos - 1  # Adjust index for 0-based
        x2, y2 = self.player2_pos - 1
        grid[5 - y1 - 1][x1] = "P1"
        grid[5 - y2 - 1][x2] = "P2"

        # Build grid string
        grid_str = ""
        for row in grid:
            grid_str += "+----+----+----+----+----+\n"
            grid_str += "|"
            for cell in row:
                grid_str += f" {cell} |"
            grid_str += "\n"
        grid_str += "+----+----+----+----+----+\n"
        return grid_str

    def valid_moves(self):
        valid_actions = []
        for action in range(self.action_space.n):
            direction, num_spaces = self._action_to_movement(action)
            if self.current_player == 1:
                current_pos = self.player1_pos.copy()
                opponent_pos = self.player2_pos.copy()
            else:
                current_pos = self.player2_pos.copy()
                opponent_pos = self.player1_pos.copy()

            # Compute new position
            new_pos = current_pos.copy()
            if direction == "up":
                new_pos[1] += num_spaces
            elif direction == "down":
                new_pos[1] -= num_spaces
            elif direction == "left":
                new_pos[0] -= num_spaces
            elif direction == "right":
                new_pos[0] += num_spaces

            if self._is_valid_move(current_pos, new_pos, opponent_pos):
                valid_actions.append(action)
        return valid_actions

    def _action_to_movement(self, action):
        # Map action number to direction and spaces
        action_mapping = {
            0: ("up", 1),
            1: ("up", 2),
            2: ("down", 1),
            3: ("down", 2),
            4: ("left", 1),
            5: ("left", 2),
            6: ("right", 1),
            7: ("right", 2),
        }
        return action_mapping[action]

    def _is_valid_move(self, current_pos, new_pos, opponent_pos):
        # Check if new position is within grid bounds
        if not all(1 <= coord <= 5 for coord in new_pos):
            return False

        # Check for linear movement (no diagonal moves)
        if (current_pos[0] != new_pos[0]) and (current_pos[1] != new_pos[1]):
            return False

        # Check for passing through opponent's position
        path_positions = self._get_path_positions(current_pos, new_pos)
        for pos in path_positions:
            if np.array_equal(pos, opponent_pos):
                return False  # Cannot pass through opponent

        return True

    def _get_path_positions(self, current_pos, new_pos):
        # Get positions between current_pos and new_pos (excluding both)
        positions = []
        if current_pos[0] == new_pos[0]:
            # Vertical movement
            step = 1 if new_pos[1] > current_pos[1] else -1
            for y in range(current_pos[1] + step, new_pos[1], step):
                positions.append(np.array([current_pos[0], y]))
        elif current_pos[1] == new_pos[1]:
            # Horizontal movement
            step = 1 if new_pos[0] > current_pos[0] else -1
            for x in range(current_pos[0] + step, new_pos[0], step):
                positions.append(np.array([x, current_pos[1]]))
        return positions

    def _get_obs(self):
        return np.concatenate([self.player1_pos, self.player2_pos])
