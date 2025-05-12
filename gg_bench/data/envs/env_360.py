import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.copy(self.board), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return np.copy(self.board), 0, True, False, {}

        row = action // 5
        col = action % 5

        # Check if the action is within bounds
        if not (0 <= row < 5 and 0 <= col < 5):
            reward = -10
            self.done = True
            return np.copy(self.board), reward, True, False, {}

        # Check if cell is empty
        if self.board[row][col] != 0:
            reward = -10
            self.done = True
            return np.copy(self.board), reward, True, False, {}

        # Check if the cell is adjacent to any of the opponent's markers
        invalid_move = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r = row + dr
                c = col + dc
                if 0 <= r < 5 and 0 <= c < 5:
                    if self.board[r][c] == -self.current_player:
                        invalid_move = True
                        break
            if invalid_move:
                break

        if invalid_move:
            reward = -10  # Invalid move
            self.done = True
            return np.copy(self.board), reward, True, False, {}

        # Valid move, place the marker
        self.board[row][col] = self.current_player

        # Check if the opponent has any valid moves
        opponent_has_moves = self.check_opponent_moves()

        if not opponent_has_moves:
            # Current player wins
            reward = 1
            self.done = True
            return np.copy(self.board), reward, True, False, {}
        else:
            # Switch to opponent
            self.current_player *= -1
            return np.copy(self.board), 0, False, False, {}

    def check_opponent_moves(self):
        opponent = -self.current_player
        for r in range(5):
            for c in range(5):
                if self.board[r][c] == 0:
                    # Check if the cell is adjacent to any of the current player's markers
                    adjacent_to_current = False
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            rr = r + dr
                            cc = c + dc
                            if 0 <= rr < 5 and 0 <= cc < 5:
                                if self.board[rr][cc] == self.current_player:
                                    adjacent_to_current = True
                                    break
                        if adjacent_to_current:
                            break
                    if not adjacent_to_current:
                        # Opponent has at least one valid move
                        return True
        return False  # Opponent has no valid moves

    def render(self):
        board_str = "   1   2   3   4   5\n"
        board_str += " +---+---+---+---+---+\n"
        for i in range(5):
            row_str = f"{i+1}|"
            for j in range(5):
                cell = self.board[i][j]
                if cell == 1:
                    row_str += " X |"
                elif cell == -1:
                    row_str += " O |"
                else:
                    row_str += "   |"
            board_str += row_str + "\n"
            board_str += " +---+---+---+---+---+\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        for action in range(25):
            row = action // 5
            col = action % 5
            if self.board[row][col] != 0:
                continue
            # Check if the cell is adjacent to any of the opponent's markers
            adjacent_to_opponent = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r = row + dr
                    c = col + dc
                    if 0 <= r < 5 and 0 <= c < 5:
                        if self.board[r][c] == -self.current_player:
                            adjacent_to_opponent = True
                            break
                if adjacent_to_opponent:
                    break
            if not adjacent_to_opponent:
                valid_actions.append(action)
        return valid_actions
