import numpy as np
from gymnasium import Env, spaces


class CustomEnv(Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define positions excluding the center square (1,1)
        self.place_positions = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ]
        # Map action IDs to place positions
        self.place_action_mapping = {
            action_id: pos for action_id, pos in enumerate(self.place_positions)
        }
        # Define movement directions: Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Start action IDs for move actions after place actions
        action_id = len(self.place_positions)
        self.move_action_mapping = {}
        # Generate move actions between adjacent cells
        for from_row in range(3):
            for from_col in range(3):
                from_pos = (from_row, from_col)
                for dr, dc in directions:
                    to_row, to_col = from_row + dr, from_col + dc
                    if 0 <= to_row < 3 and 0 <= to_col < 3:
                        to_pos = (to_row, to_col)
                        self.move_action_mapping[action_id] = (from_pos, to_pos)
                        action_id += 1

        # Total number of actions
        self.total_actions = action_id
        self.action_space = spaces.Discrete(self.total_actions)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # 1 for Player A, -1 for Player B
        self.tokens_to_place = {1: 4, -1: 4}  # Tokens left for each player
        self.done = False
        return self.board.copy(), {}  # Observation, info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        reward = 0  # Default reward
        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self.board.copy(), -10, True, False, {}

        if action in self.place_action_mapping:
            # Place action
            row, col = self.place_action_mapping[action]
            self.board[row, col] = self.current_player
            self.tokens_to_place[self.current_player] -= 1
        elif action in self.move_action_mapping:
            # Move action
            from_pos, to_pos = self.move_action_mapping[action]
            from_row, from_col = from_pos
            to_row, to_col = to_pos
            self.board[from_row, from_col] = 0
            self.board[to_row, to_col] = self.current_player
        else:
            # Invalid action ID (should not occur)
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Check for victory
        if self.board[1, 1] == self.current_player:
            self.done = True
            reward = 1  # Current player wins
            return self.board.copy(), reward, True, False, {}

        # Switch current player
        self.current_player *= -1
        return (
            self.board.copy(),
            reward,
            False,
            False,
            {},
        )  # Observation, reward, done, truncated, info

    def render(self):
        board_str = ""
        for row in range(3):
            board_str += "Row {}  ".format(row)
            for col in range(3):
                cell = self.board[row, col]
                if cell == 1:
                    board_str += "A "
                elif cell == -1:
                    board_str += "B "
                else:
                    board_str += "- "
            board_str += "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        if self.done:
            return valid_actions

        # Add valid place actions
        if self.tokens_to_place[self.current_player] > 0:
            for action_id, (row, col) in self.place_action_mapping.items():
                if self.board[row, col] == 0:
                    valid_actions.append(action_id)

        # Add valid move actions
        for action_id, (from_pos, to_pos) in self.move_action_mapping.items():
            from_row, from_col = from_pos
            to_row, to_col = to_pos
            if (
                self.board[from_row, from_col] == self.current_player
                and self.board[to_row, to_col] == 0
            ):
                valid_actions.append(action_id)
        return valid_actions
