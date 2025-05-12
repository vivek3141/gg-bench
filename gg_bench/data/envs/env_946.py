import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 90 actions (9 cells * (no swap + 9 possible swap cells))
        self.action_space = spaces.Discrete(90)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        # Initialize the board
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # Player 1 starts (1 for 'X', -1 for 'O')
        self.done = False
        self.last_player = None
        return self.board.copy(), {}

    def step(self, action):
        if self.done:
            return self.board.copy(), -10, True, False, {}

        # Decode the action
        place_cell = action // 10
        swap_cell_index = (action % 10) - 1  # -1 indicates no swap

        # Check if place_cell is valid
        if self.board[place_cell] != 0:
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Place the marker
        self.board[place_cell] = self.current_player

        # Handle swap option
        if swap_cell_index != -1:
            swap_cell = swap_cell_index

            # Check if swap_cell is valid
            if swap_cell < 0 or swap_cell > 8 or swap_cell == place_cell:
                self.done = True
                return self.board.copy(), -10, True, False, {}
            # Check if swap_cell is adjacent
            if not self._is_adjacent(place_cell, swap_cell):
                self.done = True
                return self.board.copy(), -10, True, False, {}
            # Check if swap_cell has opponent's marker
            if self.board[swap_cell] != -self.current_player:
                self.done = True
                return self.board.copy(), -10, True, False, {}
            # Perform the swap
            self.board[swap_cell] = self.current_player

        # Check if the game is over
        if np.all(self.board != 0):
            self.done = True
            # Count markers
            player1_markers = np.sum(self.board == 1)
            player2_markers = np.sum(self.board == -1)
            # Determine winner
            if player1_markers > player2_markers:
                winner = 1
            elif player2_markers > player1_markers:
                winner = -1
            else:
                # Tie-breaker: player who did NOT take the last turn wins
                winner = -self.current_player

            # Set reward
            if winner == self.current_player:
                reward = 1
            else:
                reward = -1
            return self.board.copy(), reward, True, False, {}
        else:
            # Game continues
            reward = 0
            self.last_player = self.current_player
            self.current_player *= -1
            return self.board.copy(), reward, False, False, {}

    def render(self):
        board_str = "\n"
        for i in range(3):
            row = ""
            for j in range(3):
                cell = self.board[i * 3 + j]
                if cell == 1:
                    row += " X "
                elif cell == -1:
                    row += " O "
                else:
                    row += " - "
                if j < 2:
                    row += "|"
            board_str += row
            if i < 2:
                board_str += "\n-----------\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        for place_cell in range(9):
            if self.board[place_cell] == 0:
                # No swap option
                action_no_swap = place_cell * 10 + 0
                valid_actions.append(action_no_swap)
                # Swap options
                for swap_cell in range(9):
                    if (
                        swap_cell != place_cell
                        and self.board[swap_cell] == -self.current_player
                    ):
                        if self._is_adjacent(place_cell, swap_cell):
                            action_with_swap = place_cell * 10 + swap_cell + 1
                            valid_actions.append(action_with_swap)
        return valid_actions

    def _is_adjacent(self, cell1, cell2):
        # Define adjacency (up, down, left, right)
        row1, col1 = divmod(cell1, 3)
        row2, col2 = divmod(cell2, 3)
        return (abs(row1 - row2) + abs(col1 - col2)) == 1
