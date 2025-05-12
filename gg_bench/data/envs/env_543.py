import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0-24 -> Place marker at cell (row, col)
        #          25-49 -> Remove marker from cell (row, col)
        self.action_space = spaces.Discrete(50)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(5, 5), dtype=np.int8)

        # Initialize the board and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((5, 5), dtype=np.int8)
        self.current_player = 1  # Player A: 1, Player B: -1
        self.done = False
        self.first_move = {
            1: True,
            -1: True,
        }  # Track if it's the first move for each player
        return self.board.copy(), {}

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Determine action type and target cell
        if 0 <= action <= 24:
            action_type = "place"
            cell_index = action
        elif 25 <= action <= 49:
            action_type = "remove"
            cell_index = action - 25
        else:
            # Invalid action index
            return self.board.copy(), -10, True, False, {}

        row, col = divmod(cell_index, 5)
        cell_value = self.board[row, col]

        if action_type == "place":
            # Check if cell is empty
            if cell_value != 0:
                return self.board.copy(), -10, True, False, {}

            if self.first_move[self.current_player]:
                # First move: Must place on an empty edge cell
                if row == 0 or row == 4 or col == 0 or col == 4:
                    self.board[row, col] = self.current_player
                    self.first_move[self.current_player] = False
                else:
                    return self.board.copy(), -10, True, False, {}
            else:
                # Subsequent moves: Must place adjacent to existing marker
                if self.is_adjacent_to_own_marker(row, col):
                    self.board[row, col] = self.current_player
                else:
                    return self.board.copy(), -10, True, False, {}

            # Check for win condition
            if row == 2 and col == 2:
                self.done = True
                return self.board.copy(), 1, True, False, {}

        elif action_type == "remove":
            # Can only remove own marker
            if cell_value == self.current_player:
                self.board[row, col] = 0
            else:
                return self.board.copy(), -10, True, False, {}
        else:
            # Invalid action type
            return self.board.copy(), -10, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        return self.board.copy(), 0, False, False, {}

    def is_adjacent_to_own_marker(self, row, col):
        # Check all adjacent cells
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = row + dr, col + dc
                if 0 <= r < 5 and 0 <= c < 5:
                    if (dr != 0 or dc != 0) and self.board[r, c] == self.current_player:
                        return True
        return False

    def render(self):
        board_str = ""
        symbols = {0: ".", 1: "A", -1: "B"}
        for row in range(5):
            for col in range(5):
                if row == 2 and col == 2:
                    cell = "T"  # Treasure cell
                else:
                    cell = symbols[self.board[row, col]]
                board_str += f"{cell} "
            board_str += "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        for action in range(self.action_space.n):
            if self.is_valid_action(action):
                valid_actions.append(action)
        return valid_actions

    def is_valid_action(self, action):
        if self.done:
            return False

        if 0 <= action <= 24:
            action_type = "place"
            cell_index = action
        elif 25 <= action <= 49:
            action_type = "remove"
            cell_index = action - 25
        else:
            return False

        row, col = divmod(cell_index, 5)
        cell_value = self.board[row, col]

        if action_type == "place":
            if cell_value != 0:
                return False
            if self.first_move[self.current_player]:
                return row == 0 or row == 4 or col == 0 or col == 4
            else:
                return self.is_adjacent_to_own_marker(row, col)
        elif action_type == "remove":
            return cell_value == self.current_player
        else:
            return False
