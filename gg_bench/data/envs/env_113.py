import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 64 possible grid cells (0 to 63)
        self.action_space = spaces.Discrete(64)

        # Define observation space: an array of 64 values
        # -1: cell not guessed yet
        # 0 to 14: feedback distances
        self.observation_space = spaces.Box(
            low=-1, high=14, shape=(64,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = seeding.np_random(seed)

        # Initialize the grid to -1 (unguessed)
        self.grid = np.full(64, -1, dtype=np.int32)
        self.current_player = 1  # Player 1: 1, Player 2: -1
        self.done = False

        # Randomly place the treasure
        self.treasure_location = self.np_random.randint(
            0, 64
        )  # Random cell from 0 to 63

        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):

        if self.done:
            return self.grid.copy(), 0, True, False, {}

        # Validate action
        if not self.action_space.contains(action):
            # Invalid action (outside 0 to 63)
            return (
                self.grid.copy(),
                -10,
                True,
                False,
                {"invalid_action": True, "message": "Invalid action"},
            )

        if self.grid[action] != -1:
            # Repeated guess
            return (
                self.grid.copy(),
                -10,
                False,
                False,
                {"invalid_action": True, "message": "Cell already guessed"},
            )

        # Valid move
        # Convert action index to row and column
        row = action // 8  # 0-based row index
        col = action % 8  # 0-based column index

        # Treasure coordinates
        treasure_row = self.treasure_location // 8
        treasure_col = self.treasure_location % 8

        # Calculate Manhattan distance
        distance = abs(row - treasure_row) + abs(col - treasure_col)

        # Update grid with feedback distance
        self.grid[action] = distance

        if distance == 0:
            # Treasure found
            self.done = True
            reward = 1  # Reward for winning
            return self.grid.copy(), reward, True, False, {}
        else:
            # Valid move but did not find the treasure
            reward = -10  # Negative reward per prompt
            self.current_player *= -1  # Switch player
            return self.grid.copy(), reward, False, False, {}

    def render(self):
        # Generate a string representation of the grid
        grid_str = ""
        for row in range(8):
            grid_str += "+----" * 8 + "+\n"
            for col in range(8):
                idx = row * 8 + col
                value = self.grid[idx]
                if value == -1:
                    cell_str = "    "  # Empty cell
                else:
                    cell_str = f"{value:2d}  "  # Distance value
                grid_str += "| " + cell_str
            grid_str += "|\n"
        grid_str += "+----" * 8 + "+\n"
        return grid_str

    def valid_moves(self):
        return [i for i in range(64) if self.grid[i] == -1]
