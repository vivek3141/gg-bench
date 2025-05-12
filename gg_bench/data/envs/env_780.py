import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space

        # Action space: 25 cells * 9 numbers = 225 possible actions
        self.action_space = spaces.Discrete(225)

        # Observation space: a 5x5 grid, each cell holds [player_id, number]
        # player_id: 0 (empty), 1 (Player 1), 2 (Player 2)
        # number: 0 (empty), or 1-9
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(25, 2), dtype=np.int32
        )

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the board to empty
        self.board = np.zeros((25, 2), dtype=np.int32)
        # Set current player: 1 (Player 1) starts
        self.current_player = 1
        # Game is not over
        self.done = False
        return self.board.copy(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}  # Game already over

        # Translate action into cell index and number
        cell_index = action // 9  # Cell index from 0 to 24
        number = (action % 9) + 1  # Number from 1 to 9

        # Check if cell is empty
        if self.board[cell_index, 0] != 0:
            self.done = True
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Place the number
        self.board[cell_index, 0] = self.current_player  # Mark the cell with player ID
        self.board[cell_index, 1] = number  # Place the number

        # Check for win condition
        has_won = self.check_win()

        if has_won:
            self.done = True
            return self.board.copy(), 1, True, False, {}  # Current player wins

        # Check for draw (no valid moves left)
        if np.all(self.board[:, 0] != 0):
            self.done = True
            return self.board.copy(), 0, True, False, {}  # Game drawn

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1
        return self.board.copy(), 0, False, False, {}  # Continue game

    def render(self):
        # Return a visual representation of the board as a string
        board_str = ""
        for i in range(5):
            row_str = ""
            for j in range(5):
                cell_index = i * 5 + j
                player_id = self.board[cell_index, 0]
                number = self.board[cell_index, 1]
                if player_id == 0:
                    cell_str = " . "
                else:
                    symbol = "X" if player_id == 1 else "O"
                    cell_str = f"{symbol}{number}"
                row_str += f"{cell_str:>5}"
            board_str += row_str + "\n"
        return board_str

    def valid_moves(self):
        # Return a list of valid action indices
        valid_moves = [
            cell_index * 9 + number_offset
            for cell_index in range(25)
            if self.board[cell_index, 0] == 0
            for number_offset in range(9)
        ]
        return valid_moves

    def check_win(self):
        # Check if the current player has a winning path
        player_id = self.current_player
        board = self.board.reshape(5, 5, 2)  # Reshape to 5x5 grid

        # Get grids of player's cells and the numbers
        player_grid = board[:, :, 0] == player_id
        number_grid = board[:, :, 1]

        # Identify starting and ending positions based on player
        if player_id == 1:
            # Player 1 aims to connect left to right
            start_positions = [(i, 0) for i in range(5) if player_grid[i, 0]]
            end_positions = [(i, 4) for i in range(5) if player_grid[i, 4]]
        else:
            # Player 2 aims to connect top to bottom
            start_positions = [(0, j) for j in range(5) if player_grid[0, j]]
            end_positions = [(4, j) for j in range(5) if player_grid[4, j]]

        if not start_positions or not end_positions:
            return False  # No possible path

        visited = set()
        queue = deque()

        # Start BFS from each starting position
        for pos in start_positions:
            r, c = pos
            initial_sum = number_grid[r, c]
            queue.append((pos, [(r, c)], initial_sum))

        while queue:
            (r, c), path, current_sum = queue.popleft()
            if current_sum > 15:
                continue  # Prune paths exceeding sum 15
            if (r, c) in end_positions and current_sum == 15:
                return True  # Winning path found

            # Explore adjacent cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # Skip the current cell
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 5 and 0 <= nc < 5:
                        if (nr, nc) not in path and player_grid[nr, nc]:
                            new_sum = current_sum + number_grid[nr, nc]
                            if new_sum <= 15:
                                queue.append(((nr, nc), path + [(nr, nc)], new_sum))
        return False  # No winning path found
