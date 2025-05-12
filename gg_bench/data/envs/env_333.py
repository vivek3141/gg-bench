import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(25 + 25*25) = 650 actions
        self.action_space = spaces.Discrete(650)

        # Observation space: 25 cells with values -1, 0, 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(25,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(25, dtype=np.int8)
        self.current_player = 1  # 1 for 'X', -1 for 'O'
        self.game_phase = "normal"
        self.done = False
        self.winner = None
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions or self.done:
            # Invalid move
            reward = -10
            terminated = True
            self.done = True
            return self.board.copy(), reward, terminated, truncated, info

        if action < 25:
            # Normal play
            to_cell = action
            # Place marker
            self.board[to_cell] = self.current_player
        else:
            # Sudden-death
            action_index = action - 25
            from_cell = action_index // 25
            to_cell = action_index % 25
            # Move marker
            self.board[from_cell] = 0
            self.board[to_cell] = self.current_player

        # Check for win
        if self.check_win():
            reward = 1
            terminated = True
            self.done = True
            self.winner = self.current_player
            return self.board.copy(), reward, terminated, truncated, info

        # Check for full board or no valid moves in sudden-death
        if self.game_phase == "normal" and np.all(self.board != 0):
            # Enter sudden-death phase
            self.game_phase = "sudden-death"
        elif self.game_phase == "sudden-death" and len(self.valid_moves()) == 0:
            # No more moves possible
            terminated = True
            self.done = True
            reward = 0
            return self.board.copy(), reward, terminated, truncated, info

        # Switch player
        self.current_player *= -1

        return self.board.copy(), reward, terminated, truncated, info

    def render(self):
        board_str = ""
        for i in range(5):
            row = "|"
            for j in range(5):
                cell = self.board[i * 5 + j]
                if cell == 1:
                    row += " X |"
                elif cell == -1:
                    row += " O |"
                else:
                    row += "   |"
            board_str += row + "\n" + "-" * 21 + "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        if self.done:
            return valid_actions
        if self.game_phase == "normal":
            # Valid actions are empty cells
            empty_cells = np.where(self.board == 0)[0]
            valid_actions = empty_cells.tolist()
        else:
            # Sudden-death
            # Find own markers
            own_cells = np.where(self.board == self.current_player)[0]
            empty_cells = np.where(self.board == 0)[0]
            for from_cell in own_cells:
                for to_cell in empty_cells:
                    action = 25 + from_cell * 25 + to_cell
                    valid_actions.append(action)
        return valid_actions

    def check_win(self):
        # Use BFS to find connected components
        visited = np.zeros(25, dtype=bool)
        for idx in np.where(self.board == self.current_player)[0]:
            if not visited[idx]:
                group = self.bfs(idx, visited)
                if len(group) == 4:
                    return True  # Win condition met
        return False

    def bfs(self, start_idx, visited):
        queue = [start_idx]
        visited[start_idx] = True
        group = [start_idx]
        while queue:
            idx = queue.pop(0)
            neighbors = self.get_neighbors(idx)
            for neighbor in neighbors:
                if (
                    not visited[neighbor]
                    and self.board[neighbor] == self.current_player
                ):
                    visited[neighbor] = True
                    queue.append(neighbor)
                    group.append(neighbor)
        return group

    def get_neighbors(self, idx):
        neighbors = []
        row = idx // 5
        col = idx % 5
        # Up
        if row > 0:
            neighbors.append((row - 1) * 5 + col)
        # Down
        if row < 4:
            neighbors.append((row + 1) * 5 + col)
        # Left
        if col > 0:
            neighbors.append(row * 5 + (col - 1))
        # Right
        if col < 4:
            neighbors.append(row * 5 + (col + 1))
        return neighbors
