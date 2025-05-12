import numpy as np
import gymnasium as gym
from gymnasium import spaces
import gymnasium.utils.seeding


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 4 possible actions: up, down, left, right
        self.action_space = spaces.Discrete(4)

        # The observation is a 5x5 grid with values in {-1, 0, 1, 2}
        # -1: Player 2
        # 0: Empty cell
        # 1: Player 1
        # 2: Wall
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=np.int8)

        # Initialize the grid and other variables
        self.grid = None
        self.current_player = None
        self.player_positions = None
        self.walls = None
        self.done = None
        self.np_random = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gymnasium.utils.seeding.np_random(seed)

        # Initialize grid to zeros
        self.grid = np.zeros((5, 5), dtype=np.int8)

        # Record positions of walls
        self.walls = []

        # Place walls
        self._place_walls()

        # Place players in starting positions
        self._place_players()

        # Set current player to Player 1 (represented as 1)
        self.current_player = 1

        # Game not done
        self.done = False

        # Return observation and info
        return self.grid.copy(), {}

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            terminated = True
            truncated = False
            self.done = True
            return self.grid.copy(), reward, terminated, truncated, {}

        # Map action index to movement
        move_mapping = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        # Get current position
        current_pos = self.player_positions[self.current_player]
        move = move_mapping[action]
        new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

        # Update grid
        self.grid[current_pos] = 0  # Clear old position
        self.grid[new_pos] = self.current_player  # Set new position
        self.player_positions[self.current_player] = new_pos  # Update position

        # Check for victory
        if self._check_win(new_pos):
            reward = 1
            terminated = True
            truncated = False
            self.done = True
            return self.grid.copy(), reward, terminated, truncated, {}

        # No reward for valid move
        reward = 0
        terminated = False
        truncated = False

        # Switch to other player
        self.current_player *= -1

        # If the next player has no valid moves, they forfeit their turn
        if not self.valid_moves():
            self.current_player *= -1  # Skip their turn
            if not self.valid_moves():
                # Neither player can move
                terminated = True
                self.done = True
                return self.grid.copy(), reward, terminated, truncated, {}

        return self.grid.copy(), reward, terminated, truncated, {}

    def render(self):
        symbols = {0: ".", 1: "P1", -1: "P2", 2: "X"}
        grid_str = "   " + "  ".join(str(i + 1) for i in range(5)) + "\n"
        grid_str += "  " + "---" * 5 + "\n"
        for i in range(5):
            row = [symbols[self.grid[i][j]] for j in range(5)]
            grid_str += f"{i+1} | " + " ".join(f"{cell:>2}" for cell in row) + "\n"
        return grid_str

    def valid_moves(self):
        # Return list of valid action indices for current player
        valid_actions = []
        move_mapping = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        current_pos = self.player_positions[self.current_player]
        for action in range(4):
            move = move_mapping[action]
            new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            if self._is_valid_position(new_pos):
                valid_actions.append(action)
        return valid_actions

    # Helper methods
    def _place_walls(self):
        # Randomly place 5 walls, avoiding starting positions
        possible_positions = [(i, j) for i in range(5) for j in range(5)]
        # Remove first and last columns positions (player starting columns)
        starting_positions = [(i, 0) for i in range(5)] + [(i, 4) for i in range(5)]
        possible_positions = [
            pos for pos in possible_positions if pos not in starting_positions
        ]
        # Randomly sample 5 positions for walls
        wall_positions = self.np_random.choice(
            len(possible_positions), size=5, replace=False
        )
        for idx in wall_positions:
            pos = possible_positions[idx]
            self.grid[pos] = 2  # Wall
            self.walls.append(pos)

    def _place_players(self):
        # Randomly select starting positions in respective starting columns
        # Remove any rows where walls are in starting positions
        max_attempts = 10
        for attempt in range(max_attempts):
            p1_starting_rows = [i for i in range(5) if self.grid[i][0] == 0]
            p2_starting_rows = [i for i in range(5) if self.grid[i][4] == 0]

            # If no starting positions available, reinitialize walls
            if not p1_starting_rows or not p2_starting_rows:
                self.grid = np.zeros((5, 5), dtype=np.int8)
                self.walls = []
                self._place_walls()
                continue  # Try again

            p1_row = self.np_random.choice(p1_starting_rows)
            p2_row = self.np_random.choice(p2_starting_rows)

            self.grid[p1_row][0] = 1  # Player 1
            self.grid[p2_row][4] = -1  # Player 2

            self.player_positions = {1: (p1_row, 0), -1: (p2_row, 4)}
            return  # Players placed successfully

        # If unable to place players after max_attempts, raise an error
        raise RuntimeError(
            "Unable to place players on the grid after multiple attempts."
        )

    def _is_valid_position(self, pos):
        # Check if position is within grid bounds
        if not (0 <= pos[0] < 5 and 0 <= pos[1] < 5):
            return False
        # Check if cell is empty
        cell_value = self.grid[pos]
        if cell_value == 0:
            return True
        else:
            return False

    def _check_win(self, pos):
        # Check for victory condition
        if self.current_player == 1 and pos[1] == 4:
            return True
        if self.current_player == -1 and pos[1] == 0:
            return True
        return False
