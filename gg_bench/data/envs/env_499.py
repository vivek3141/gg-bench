import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5, 5), dtype=np.int8)
        # 4 movement directions * 23 possible block positions
        self.action_space = spaces.Discrete(92)

        # Movement directions
        self.movement_directions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1),  # Right
        }

        # Starting positions for players
        self.start_positions = {
            1: (0, 2),  # Player 1 starts at (0,2)
            2: (4, 2),  # Player 2 starts at (4,2)
        }

        # Create list of valid block positions (excluding starting positions)
        self.valid_block_positions = [
            (i, j)
            for i in range(5)
            for j in range(5)
            if (i, j) not in self.start_positions.values()
        ]
        self.num_block_positions = len(self.valid_block_positions)  # Should be 23

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((5, 5), dtype=np.int8)

        # Set starting positions
        self.board[self.start_positions[1]] = 1  # Player 1 token represented by 1
        self.board[self.start_positions[2]] = 2  # Player 2 token represented by 2

        self.current_player = 1
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        # Check if current player has moves
        valid_actions = self.valid_moves()
        if len(valid_actions) == 0:
            # Current player cannot move, they lose
            self.done = True
            return self.board.copy(), -1, True, False, {}

        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Valid action, proceed
        move_direction = action // self.num_block_positions
        block_position_index = action % self.num_block_positions

        move_delta = self.movement_directions[move_direction]
        block_position = self.valid_block_positions[block_position_index]

        # Get current player position
        current_pos = np.argwhere(self.board == self.current_player)[0]

        # Compute new position after move
        new_pos = current_pos + move_delta

        # Move is valid, perform the move
        self.board[current_pos[0], current_pos[1]] = 0
        self.board[new_pos[0], new_pos[1]] = self.current_player

        # Place the block
        self.board[block_position[0], block_position[1]] = (
            -1
        )  # Block is represented by -1

        # Check victory conditions
        # 1. Current player reaches opponent's starting position
        opponent_start_pos = self.start_positions[3 - self.current_player]
        if (new_pos[0], new_pos[1]) == opponent_start_pos:
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # 2. Check if opponent has any legal moves
        opponent_has_moves = self.check_player_has_moves(3 - self.current_player)
        if not opponent_has_moves:
            # Opponent has no moves, current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch to next player
        self.current_player = 3 - self.current_player
        return self.board.copy(), 0, False, False, {}

    def check_player_has_moves(self, player):
        # Find player's token position
        player_pos = np.argwhere(self.board == player)
        if len(player_pos) == 0:
            # Player's token not found
            return False
        player_pos = player_pos[0]

        # For each possible movement direction
        for move_delta in self.movement_directions.values():
            new_pos = player_pos + move_delta
            if 0 <= new_pos[0] < 5 and 0 <= new_pos[1] < 5:
                cell_value = self.board[new_pos[0], new_pos[1]]
                if cell_value == 0:
                    # Empty space, move possible
                    return True
        # No moves possible
        return False

    def render(self):
        board_str = ""
        for i in range(5):
            board_str += "|"
            for j in range(5):
                cell = self.board[i, j]
                if cell == 0:
                    board_str += "   |"
                elif cell == -1:
                    board_str += " X |"
                elif cell == 1:
                    board_str += "P1 |"
                elif cell == 2:
                    board_str += "P2 |"
            board_str += "\n"
        return board_str

    def valid_moves(self):
        if self.done:
            return []

        valid_actions = []
        # Get current player's token position
        current_pos = np.argwhere(self.board == self.current_player)
        if len(current_pos) == 0:
            return []
        current_pos = current_pos[0]

        # For each possible movement direction
        for move_dir, move_delta in self.movement_directions.items():
            new_pos = current_pos + move_delta
            if 0 <= new_pos[0] < 5 and 0 <= new_pos[1] < 5:
                cell_value = self.board[new_pos[0], new_pos[1]]
                if cell_value == 0:
                    # Movement possible
                    # For each valid block position
                    for block_index, block_pos in enumerate(self.valid_block_positions):
                        # Check if block position is empty
                        if self.board[block_pos[0], block_pos[1]] == 0:
                            # Create action index
                            action_index = (
                                move_dir * self.num_block_positions + block_index
                            )
                            valid_actions.append(action_index)
        return valid_actions
