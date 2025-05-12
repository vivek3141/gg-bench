import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Action space is Discrete(25) - move to any square (coordinates from 0 to 24)
        self.action_space = spaces.Discrete(25)

        # Observation space is a 5x5 grid with integer values from -2 to 2
        # -2: Blocked by Player 2
        # -1: Blocked by Player 1
        # 0: Empty square
        # 1: Player 1's marker
        # 2: Player 2's marker

        self.observation_space = spaces.Box(low=-2, high=2, shape=(5, 5), dtype=np.int8)

        # Initialize the grid and other variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the grid
        self.grid = np.zeros((5, 5), dtype=np.int8)

        # Set starting positions
        self.grid[0, 0] = 1  # Player 1 starts at top-left corner (A1)
        self.grid[4, 4] = 2  # Player 2 starts at bottom-right corner (E5)

        # Record players' positions
        self.player_positions = {1: (0, 0), 2: (4, 4)}

        # Record squares each player has blocked
        self.blocked_by_player = {1: set(), 2: set()}

        # Current player (1 or 2)
        self.current_player = 1

        # Flag to indicate if the game is done
        self.done = False

        return self.grid.copy(), {}  # Return observation and info

    def step(self, action):
        # Check if the game is already over
        if self.done:
            return self.grid.copy(), -10, True, False, {}

        # At the start of the player's turn, check if they have any valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Current player cannot move, they lose
            self.done = True
            reward = -10  # Current player loses
            return self.grid.copy(), reward, True, False, {}

        # Validate that the action is in the list of valid actions
        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10  # Invalid move, current player loses
            return self.grid.copy(), reward, True, False, {}

        # Convert action index to grid coordinates
        target_row = action // 5  # integer division
        target_col = action % 5

        # Get current player position
        curr_row, curr_col = self.player_positions[self.current_player]

        # Move player's marker
        self.grid[curr_row, curr_col] = 0  # Empty the current square

        # Block the square the player just left (blocked for opponent)
        self.grid[curr_row, curr_col] = -self.current_player

        # Record that the player has blocked this square
        self.blocked_by_player[self.current_player].add((curr_row, curr_col))

        # Update player's position
        self.player_positions[self.current_player] = (target_row, target_col)

        # Place player's marker on the target square
        self.grid[target_row, target_col] = self.current_player

        # Check for victory condition
        opponent_start_positions = {1: (4, 4), 2: (0, 0)}
        if (target_row, target_col) == opponent_start_positions[self.current_player]:
            # Current player has reached opponent's starting square
            self.done = True
            reward = 1  # Current player wins
            return self.grid.copy(), reward, True, False, {}

        # Switch player
        previous_player = self.current_player
        self.current_player = 3 - self.current_player

        # Check if the new current player can move
        valid_actions = self.valid_moves()
        if not valid_actions:
            # New current player cannot move, they lose
            self.done = True
            # Previous player wins
            reward = 1  # Previous player wins
            return self.grid.copy(), reward, True, False, {}

        # Since the move was valid and game continues, reward is -10
        reward = -10
        return self.grid.copy(), reward, False, False, {}

    def get_valid_moves(self, player):
        # Return a list of valid moves (row, col) for the given player
        moves = []
        curr_row, curr_col = self.player_positions[player]

        # Possible moves: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            nr, nc = curr_row + dr, curr_col + dc
            if 0 <= nr < 5 and 0 <= nc < 5:
                # Check if target square is occupied by opponent's marker
                if self.grid[nr, nc] == 3 - player:
                    continue

                # Check if target square is blocked by opponent
                if self.grid[nr, nc] == -(3 - player):
                    continue

                # Check if player has previously blocked this square
                if (nr, nc) in self.blocked_by_player[player]:
                    continue

                # Check if target square is own blocked square
                if self.grid[nr, nc] == -player:
                    # Cannot end turn here
                    continue

                # Valid move
                moves.append((nr, nc))
        return moves

    def valid_moves(self):
        # Return a list of action indices that are valid moves for the current player
        moves = self.get_valid_moves(self.current_player)
        action_indices = [r * 5 + c for r, c in moves]
        return action_indices

    def render(self):
        # Return a visual representation of the grid as a string
        grid_str = "   A   B   C   D   E\n"
        for i in range(5):
            grid_str += f"{i+1} "
            for j in range(5):
                cell = self.grid[i, j]
                if cell == 1:
                    grid_str += "[P1]"
                elif cell == 2:
                    grid_str += "[P2]"
                elif cell == -1 or cell == -2:
                    grid_str += "[XX]"  # Blocked square
                else:
                    grid_str += "[--]"
            grid_str += "\n"
        return grid_str
