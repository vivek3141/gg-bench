import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 33 possible actions:
        # Actions 0-8: Place a blocker at cell 1-9
        # Actions 9-32: Move a blocker from one cell to an adjacent cell
        self.action_space = spaces.Discrete(33)

        # Observation space is an array of 11 elements:
        # Elements 0-8: Cells 1-9
        #   0: Empty
        #   1: Player 1 base
        #   2: Player 1 blocker
        #   -1: Player 2 base
        #   -2: Player 2 blocker
        # Element 9: Player 1 blockers in reserve (0-3)
        # Element 10: Player 2 blockers in reserve (0-3)
        self.observation_space = spaces.Box(low=-2, high=3, shape=(11,), dtype=np.int8)

        # Initialize action mappings
        self._init_action_mappings()

        # Initialize the environment
        self.reset()

    def _init_action_mappings(self):
        # Mapping from action indices to actions
        # Actions 0-8: Place blocker at cell (index + 1)
        # Actions 9-32: Movement actions from_cell to to_cell
        self.movement_actions = {
            9: (1, 2),
            10: (1, 4),
            11: (2, 1),
            12: (2, 3),
            13: (2, 5),
            14: (3, 2),
            15: (3, 6),
            16: (4, 1),
            17: (4, 5),
            18: (4, 7),
            19: (5, 2),
            20: (5, 4),
            21: (5, 6),
            22: (5, 8),
            23: (6, 3),
            24: (6, 5),
            25: (6, 9),
            26: (7, 4),
            27: (7, 8),
            28: (8, 5),
            29: (8, 7),
            30: (8, 9),
            31: (9, 6),
            32: (9, 8),
        }

        # Reverse mapping for valid_moves() checks
        self.cell_adjacency = {
            1: [2, 4],
            2: [1, 3, 5],
            3: [2, 6],
            4: [1, 5, 7],
            5: [2, 4, 6, 8],
            6: [3, 5, 9],
            7: [4, 8],
            8: [5, 7, 9],
            9: [6, 8],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.board[0] = 1  # Player 1 base at cell 1
        self.board[8] = -1  # Player 2 base at cell 9
        self.blockers_in_reserve = [3, 3]  # [Player 1, Player 2] blockers
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            return self._get_observation(), -10, True, False, {}

        reward = 0

        # Perform the action
        if action <= 8:
            # Placement action
            cell = action  # Cell index (0-8)
            self._place_blocker(cell)
        else:
            # Movement action
            from_cell, to_cell = self.movement_actions[action]
            self._move_blocker(from_cell - 1, to_cell - 1)

        # Check for win condition
        if self._check_win():
            reward = 1
            self.done = True
        else:
            # Switch player
            self.current_player *= -1

        return self._get_observation(), reward, self.done, False, {}

    def _place_blocker(self, cell):
        # Place a blocker at the specified cell
        self.board[cell] = 2 if self.current_player == 1 else -2
        player_index = 0 if self.current_player == 1 else 1
        self.blockers_in_reserve[player_index] -= 1

    def _move_blocker(self, from_cell, to_cell):
        # Move a blocker from one cell to another
        self.board[to_cell] = self.board[from_cell]
        self.board[from_cell] = 0

    def _check_win(self):
        # Check if the current player has locked down the opponent's base
        opponent_base_cell = 8 if self.current_player == 1 else 0
        opponent_base_value = -1 if self.current_player == 1 else 1

        adjacent_cells = self.cell_adjacency[opponent_base_cell + 1]
        occupied = 0
        for adj_cell in adjacent_cells:
            cell_value = self.board[adj_cell - 1]
            # Check if the cell is occupied by the current player's blocker
            if self.current_player == 1 and cell_value == 2:
                occupied += 1
            elif self.current_player == -1 and cell_value == -2:
                occupied += 1

        if occupied == len(adjacent_cells):
            return True
        return False

    def render(self):
        board_str = "\nCurrent Board:\n"
        for i in range(3):
            row_str = ""
            for j in range(3):
                idx = i * 3 + j
                cell_value = self.board[idx]
                if cell_value == 0:
                    cell_repr = f"[{idx + 1}]"
                elif cell_value == 1:
                    cell_repr = " P1 Base "
                elif cell_value == -1:
                    cell_repr = " P2 Base "
                elif cell_value == 2:
                    cell_repr = " P1B "
                elif cell_value == -2:
                    cell_repr = " P2B "
                row_str += f"{cell_repr}\t"
            board_str += row_str + "\n"
        board_str += f"Player 1 Blockers in Reserve: {self.blockers_in_reserve[0]}\n"
        board_str += f"Player 2 Blockers in Reserve: {self.blockers_in_reserve[1]}\n"
        board_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return board_str

    def valid_moves(self):
        valid_actions = []

        if self.done:
            return valid_actions

        player_index = 0 if self.current_player == 1 else 1
        player_blocker = 2 if self.current_player == 1 else -2
        player_base = 1 if self.current_player == 1 else -1

        # Check for placement actions
        if self.blockers_in_reserve[player_index] > 0:
            for cell in range(9):
                if self.board[cell] != 0:
                    continue  # Cell is not empty
                if self._is_adjacent_to_own(cell):
                    # Action index for placement is the cell index (0-8)
                    valid_actions.append(cell)

        # Check for movement actions
        for idx in range(9):
            if self.board[idx] == player_blocker:
                from_cell = idx + 1
                for adj in self.cell_adjacency[from_cell]:
                    to_cell = adj - 1
                    if self.board[to_cell] == 0:
                        # Find action index for movement from from_cell to to_cell
                        for action_idx, (
                            f_cell,
                            t_cell,
                        ) in self.movement_actions.items():
                            if f_cell == from_cell and t_cell == adj:
                                valid_actions.append(action_idx)
        return valid_actions

    def _is_adjacent_to_own(self, cell):
        # Check if the cell is adjacent to own base or own blocker
        player_blocker = 2 if self.current_player == 1 else -2
        player_base = 1 if self.current_player == 1 else -1

        cell_number = cell + 1
        for adj in self.cell_adjacency.get(cell_number, []):
            adj_value = self.board[adj - 1]
            if adj_value == player_blocker or adj_value == player_base:
                return True
        return False

    def _get_observation(self):
        observation = np.zeros(11, dtype=np.int8)
        observation[:9] = self.board
        observation[9] = self.blockers_in_reserve[0]
        observation[10] = self.blockers_in_reserve[1]
        return observation
