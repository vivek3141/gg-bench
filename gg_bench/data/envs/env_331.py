import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is 9 discrete actions (cells 0 to 8)
        self.action_space = spaces.Discrete(9)

        # The observation will be a 9-element vector, with possible values:
        # 0: empty and unblocked
        # 1: claimed by current player
        # -1: claimed by opponent
        # 2: blocked for current player
        self.observation_space = spaces.Box(low=-1, high=2, shape=(9,), dtype=np.int8)

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)  # Empty board
        self.current_player = 1  # Player 1 starts (1 for Player 1, -1 for Player 2)

        # Initialize blocked cells for each player
        self.blocked_for_p1 = []
        self.blocked_for_p2 = []

        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # At the start of the turn, unblock cells for the current player
        if self.current_player == 1:
            self.blocked_for_p1 = []
        else:
            self.blocked_for_p2 = []

        # Check if the action is valid
        if not self._is_valid_action(action):
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Place the mark
        self.board[action] = self.current_player

        # Determine opponent
        opponent = -self.current_player

        # Apply blocking to orthogonally adjacent cells for the opponent
        blocked_cells = self._get_adjacent_cells(action)
        blocked_cells = [cell for cell in blocked_cells if self.board[cell] == 0]

        if opponent == 1:
            self.blocked_for_p1 = blocked_cells
        else:
            self.blocked_for_p2 = blocked_cells

        # Check if opponent has any valid moves
        opponent_valid_moves = self._get_valid_actions(opponent)
        if not opponent_valid_moves:
            # Current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch current player
        self.current_player = opponent

        return self._get_obs(), 0, False, False, {}

    def render(self):
        symbols = {0: " ", 1: "X", -1: "O"}
        blocked_symbols = {1: "x", -1: "o"}
        board_str = ""

        for i in range(9):
            cell = self.board[i]
            if cell == 0:
                if self.current_player == 1 and i in self.blocked_for_p1:
                    board_str += f"[{blocked_symbols[1]}]"
                elif self.current_player == -1 and i in self.blocked_for_p2:
                    board_str += f"[{blocked_symbols[-1]}]"
                else:
                    board_str += "[ ]"
            else:
                board_str += f"[{symbols[cell]}]"

            if (i + 1) % 3 == 0:
                board_str += "\n"
        return board_str

    def valid_moves(self):
        return self._get_valid_actions(self.current_player)

    def _get_obs(self):
        # Observation from the current player's perspective
        obs = np.zeros(9, dtype=np.int8)
        for i in range(9):
            if self.board[i] == self.current_player:
                obs[i] = 1  # Current player's mark
            elif self.board[i] == -self.current_player:
                obs[i] = -1  # Opponent's mark
            elif self.current_player == 1 and i in self.blocked_for_p1:
                obs[i] = 2  # Blocked for current player
            elif self.current_player == -1 and i in self.blocked_for_p2:
                obs[i] = 2  # Blocked for current player
            else:
                obs[i] = 0  # Empty and unblocked
        return obs

    def _is_valid_action(self, action):
        if action < 0 or action >= 9:
            return False
        if self.board[action] != 0:
            return False
        if self.current_player == 1 and action in self.blocked_for_p1:
            return False
        if self.current_player == -1 and action in self.blocked_for_p2:
            return False
        return True

    def _get_valid_actions(self, player):
        if player == 1:
            blocked = self.blocked_for_p1
        else:
            blocked = self.blocked_for_p2
        valid_actions = [i for i in range(9) if self.board[i] == 0 and i not in blocked]
        return valid_actions

    def _get_adjacent_cells(self, index):
        row = index // 3
        col = index % 3
        adjacent = []

        # Up
        if row > 0:
            adjacent.append(index - 3)
        # Down
        if row < 2:
            adjacent.append(index + 3)
        # Left
        if col > 0:
            adjacent.append(index - 1)
        # Right
        if col < 2:
            adjacent.append(index + 1)

        return adjacent
