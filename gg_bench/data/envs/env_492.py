import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the grid size
        self.grid_size = 5

        # Define action space:
        # 0: Move up
        # 1: Move down
        # 2: Move left
        # 3: Move right
        # 4-28: Place obstacle at cell index (i - 4)
        self.action_space = spaces.Discrete(29)

        # Define observation space
        # -2: Obstacle
        # -1: Opponent's piece
        # 0: Empty
        # 1: Current player's piece
        self.observation_space = spaces.Box(
            low=-2, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Set initial positions
        self.player_positions = {
            1: (0, 0),  # Player 1 starts at (0, 0)
            -1: (self.grid_size - 1, self.grid_size - 1),  # Player -1 starts at (4, 4)
        }
        self.board[self.player_positions[1]] = 1
        self.board[self.player_positions[-1]] = -1

        # Each player has 3 obstacles
        self.obstacles_remaining = {1: 3, -1: 3}

        # Current player (1 or -1)
        self.current_player = 1

        self.done = False

        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Copy the board to simulate changes
        board = self.board.copy()
        reward = 0
        info = {}

        # Decode action
        if action < 0 or action >= self.action_space.n:
            # Invalid action index
            return self.board.copy(), -10, True, False, {}

        if action <= 3:
            # Movement action
            direction = action
            move_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
            offset = move_offsets[direction]
            current_pos = self.player_positions[self.current_player]
            new_pos = (current_pos[0] + offset[0], current_pos[1] + offset[1])

            # Check if new position is within bounds
            if not (
                0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size
            ):
                # Invalid move
                return self.board.copy(), -10, True, False, {}

            # Check if new position is empty
            cell_value = self.board[new_pos]
            if cell_value != 0:
                # Cannot move to this cell
                return self.board.copy(), -10, True, False, {}

            # Move the piece
            board[current_pos] = 0
            board[new_pos] = 1
            self.player_positions[self.current_player] = new_pos

        else:
            # Obstacle placement action
            cell_index = action - 4
            x = cell_index % self.grid_size
            y = cell_index // self.grid_size
            target_cell = (y, x)

            # Check if player has obstacles remaining
            if self.obstacles_remaining[self.current_player] <= 0:
                # No obstacles left
                return self.board.copy(), -10, True, False, {}

            # Check if cell is empty
            if self.board[target_cell] != 0:
                # Cannot place obstacle here
                return self.board.copy(), -10, True, False, {}

            # Temporarily place the obstacle to check path
            board[target_cell] = -2

            # Check if this blocks opponent's path
            opponent = -self.current_player
            opponent_pos = self.player_positions[opponent]
            opponent_goal = self.player_positions[self.current_player]

            if not self.path_exists(board, opponent_pos, opponent_goal):
                # Cannot place obstacle here
                return self.board.copy(), -10, True, False, {}

            # Place the obstacle
            self.board[target_cell] = -2
            self.obstacles_remaining[self.current_player] -= 1

        # Check win condition
        player_pos = self.player_positions[self.current_player]
        opponent_start = self.player_positions[-self.current_player]
        if player_pos == opponent_start:
            # Current player wins
            self.done = True
            reward = 1
            return self.board.copy(), reward, True, False, {}

        # Update the board
        self.board = board

        # Switch player
        self.current_player *= -1

        return self.board.copy(), reward, False, False, info

    def render(self):
        board_str = ""
        symbol_map = {1: "P1", -1: "P2", -2: " X", 0: "  "}
        for y in range(self.grid_size):
            row = ""
            for x in range(self.grid_size):
                cell = self.board[y, x]
                symbol = symbol_map[cell]
                row += f"[{symbol}]"
            board_str += row + "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []

        # Movement actions
        move_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for i, offset in enumerate(move_offsets):
            current_pos = self.player_positions[self.current_player]
            new_pos = (current_pos[0] + offset[0], current_pos[1] + offset[1])

            # Check bounds
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                cell_value = self.board[new_pos]
                if cell_value == 0:
                    valid_actions.append(i)

        # Obstacle placement actions
        if self.obstacles_remaining[self.current_player] > 0:
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    cell_value = self.board[y, x]
                    if cell_value == 0:
                        # Temporarily place obstacle
                        board_copy = self.board.copy()
                        board_copy[y, x] = -2

                        # Check if this blocks opponent's path
                        opponent = -self.current_player
                        opponent_pos = self.player_positions[opponent]
                        opponent_goal = self.player_positions[self.current_player]

                        if self.path_exists(board_copy, opponent_pos, opponent_goal):
                            action_index = 4 + y * self.grid_size + x
                            valid_actions.append(action_index)

        return valid_actions

    def path_exists(self, board, start, goal):
        # Simple BFS to check if path exists from start to goal
        visited = np.full((self.grid_size, self.grid_size), False, dtype=bool)
        queue = deque()
        queue.append(start)
        visited[start] = True

        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            neighbors = self.get_neighbors(current, board)
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return False

    def get_neighbors(self, position, board):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for offset in directions:
            new_pos = (position[0] + offset[0], position[1] + offset[1])
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                cell_value = board[new_pos]
                if cell_value == 0 or cell_value == -self.current_player:
                    neighbors.append(new_pos)
        return neighbors
