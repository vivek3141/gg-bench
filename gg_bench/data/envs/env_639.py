import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions:
        # 0-24: Place token at cell 0-24
        # 25-124: Move token from cell (action_index-25)//4 in direction (action_index-25)%4
        self.action_space = spaces.Discrete(125)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # 1 for Shadows, -1 for Light
        self.done = False
        self.shadows_tokens = 3  # Tokens left to place for Shadows
        self.light_tokens = 3  # Tokens left to place for Light
        self.info = {}
        return self.board.copy(), self.info  # Return observation and info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.done:
            return self.board.copy(), reward, terminated, truncated, info

        # Decode the action
        if action < 0 or action >= 125:
            # Invalid action index
            reward = -10
            terminated = True
            self.done = True
            return self.board.copy(), reward, terminated, truncated, info

        if action < 25:
            # Place token at cell action (from 0 to 24)
            row = action // 5
            col = action % 5
            if self.board[row, col] != 0:
                # Invalid move
                reward = -10
                terminated = True
                self.done = True
                return self.board.copy(), reward, terminated, truncated, info

            # Check if player has tokens left to place
            if self.current_player == 1 and self.shadows_tokens <= 0:
                # Invalid move
                reward = -10
                terminated = True
                self.done = True
                return self.board.copy(), reward, terminated, truncated, info
            elif self.current_player == -1 and self.light_tokens <= 0:
                # Invalid move
                reward = -10
                terminated = True
                self.done = True
                return self.board.copy(), reward, terminated, truncated, info

            # Place the token
            self.board[row, col] = self.current_player
            if self.current_player == 1:
                self.shadows_tokens -= 1
            else:
                self.light_tokens -= 1

            # Capture opponent's adjacent tokens
            self._capture_adjacent_tokens(row, col)
        else:
            # Move token from cell to adjacent cell
            move_index = action - 25
            from_cell = move_index // 4
            direction = move_index % 4
            from_row = from_cell // 5
            from_col = from_cell % 5
            if self.board[from_row, from_col] != self.current_player:
                # Invalid move (no token of current player at from_cell)
                reward = -10
                terminated = True
                self.done = True
                return self.board.copy(), reward, terminated, truncated, info

            # Determine target cell based on direction
            if direction == 0:  # Up
                to_row, to_col = from_row - 1, from_col
            elif direction == 1:  # Down
                to_row, to_col = from_row + 1, from_col
            elif direction == 2:  # Left
                to_row, to_col = from_row, from_col - 1
            elif direction == 3:  # Right
                to_row, to_col = from_row, from_col + 1
            else:
                # Invalid direction
                reward = -10
                terminated = True
                self.done = True
                return self.board.copy(), reward, terminated, truncated, info

            # Check if target cell is within bounds and empty
            if 0 <= to_row < 5 and 0 <= to_col < 5 and self.board[to_row, to_col] == 0:
                # Move the token
                self.board[to_row, to_col] = self.current_player
                self.board[from_row, from_col] = 0

                # Capture opponent's adjacent tokens
                self._capture_adjacent_tokens(to_row, to_col)
            else:
                # Invalid move
                reward = -10
                terminated = True
                self.done = True
                return self.board.copy(), reward, terminated, truncated, info

        # Check for win condition
        shadows_score = np.sum(self.board == 1)
        light_score = np.sum(self.board == -1)
        if shadows_score >= 13 or light_score == 0:
            if self.current_player == 1:
                reward = 1
            else:
                reward = -1
            terminated = True
            self.done = True
            return self.board.copy(), reward, terminated, truncated, info
        elif light_score >= 13 or shadows_score == 0:
            if self.current_player == -1:
                reward = 1
            else:
                reward = -1
            terminated = True
            self.done = True
            return self.board.copy(), reward, terminated, truncated, info

        # Switch to the other player
        self.current_player *= -1

        return self.board.copy(), reward, terminated, truncated, info

    def render(self):
        board_str = ""
        symbol_map = {0: ".", 1: "S", -1: "L"}
        for row in self.board:
            row_str = " ".join([symbol_map[cell] for cell in row])
            board_str += row_str + "\n"
        return board_str

    def valid_moves(self):
        moves = []
        if (self.current_player == 1 and self.shadows_tokens > 0) or (
            self.current_player == -1 and self.light_tokens > 0
        ):
            # Can place tokens
            empty_cells = np.argwhere(self.board == 0)
            for cell in empty_cells:
                row, col = cell
                action = row * 5 + col
                moves.append(action)
        # Can move tokens
        player_cells = np.argwhere(self.board == self.current_player)
        for cell in player_cells:
            from_row, from_col = cell
            from_cell = from_row * 5 + from_col
            for direction in range(4):
                if direction == 0:  # Up
                    to_row, to_col = from_row - 1, from_col
                elif direction == 1:  # Down
                    to_row, to_col = from_row + 1, from_col
                elif direction == 2:  # Left
                    to_row, to_col = from_row, from_col - 1
                elif direction == 3:  # Right
                    to_row, to_col = from_row, from_col + 1
                else:
                    continue
                if (
                    0 <= to_row < 5
                    and 0 <= to_col < 5
                    and self.board[to_row, to_col] == 0
                ):
                    action = 25 + from_cell * 4 + direction
                    moves.append(action)
        return moves

    def _capture_adjacent_tokens(self, row, col):
        # Capture opponent's tokens adjacent to (row, col)
        opponent = -self.current_player
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for dr, dc in directions:
            adj_row, adj_col = row + dr, col + dc
            if (
                0 <= adj_row < 5
                and 0 <= adj_col < 5
                and self.board[adj_row, adj_col] == opponent
            ):
                self.board[adj_row, adj_col] = self.current_player
