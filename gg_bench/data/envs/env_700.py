import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation spaces
        # Action space: 6 possible actions
        # Actions:
        # 0 - Move forward
        # 1 to 5 - Place a block on Cell 1 to Cell 5 respectively
        self.action_space = spaces.Discrete(6)

        # Observation space: 5 cells with values
        # -1: Blocked cell
        # 0: Empty cell
        # 1: Player 1's piece (A)
        # 2: Player 2's piece (B)
        self.observation_space = spaces.Box(low=-1, high=2, shape=(5,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board
        # Cells are indexed from 0 to 4 (Cells 1 to 5)
        self.board = np.zeros(5, dtype=np.int8)
        self.board[0] = 1  # Player 1 starts at Cell 1
        self.board[4] = 2  # Player 2 starts at Cell 5
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, self.done, False, {}

        # Invalid move penalty and winning reward
        invalid_move_penalty = -10
        win_reward = 1

        # Move action
        if action == 0:
            # Move forward
            # Find current player's position
            player_pos = np.where(self.board == self.current_player)[0]
            if len(player_pos) == 0:
                # Player has no piece on the board; invalid state
                self.done = True
                return self.board.copy(), invalid_move_penalty, self.done, False, {}
            player_pos = player_pos[0]
            # Determine target cell
            if self.current_player == 1:
                # Player 1 moves right
                target_pos = player_pos + 1
            else:
                # Player 2 moves left
                target_pos = player_pos - 1
            # Check if target position is within bounds
            if target_pos < 0 or target_pos >= len(self.board):
                # Cannot move beyond the grid; invalid move
                self.done = True
                return self.board.copy(), invalid_move_penalty, self.done, False, {}
            # Check if target cell is empty and not blocked
            if self.board[target_pos] == 0:
                # Move is valid; update the board
                self.board[player_pos] = 0
                self.board[target_pos] = self.current_player
                # Check for winning condition
                if (self.current_player == 1 and target_pos == 4) or (
                    self.current_player == 2 and target_pos == 0
                ):
                    # Current player wins
                    self.done = True
                    return self.board.copy(), win_reward, self.done, False, {}
                else:
                    # Switch player
                    self.current_player = 2 if self.current_player == 1 else 1
                    return self.board.copy(), 0, False, False, {}
            else:
                # Target cell is occupied or blocked; invalid move
                self.done = True
                return self.board.copy(), invalid_move_penalty, self.done, False, {}
        elif 1 <= action <= 5:
            # Place block at Cell (action corresponds to cell index)
            cell_index = action - 1  # Actions 1-5 correspond to Cell 1-5 (index 0-4)
            # Validate block placement
            opponent_start = 4 if self.current_player == 1 else 0
            own_start = 0 if self.current_player == 1 else 4
            if cell_index == opponent_start or cell_index == own_start:
                # Cannot place block on opponent's or own starting cell
                self.done = True
                return self.board.copy(), invalid_move_penalty, self.done, False, {}
            if self.board[cell_index] == 0:
                # Cell is empty; place block
                self.board[cell_index] = -1  # Blocked cell
                # Switch player
                self.current_player = 2 if self.current_player == 1 else 1
                return self.board.copy(), 0, False, False, {}
            else:
                # Cell is occupied or already blocked; invalid move
                self.done = True
                return self.board.copy(), invalid_move_penalty, self.done, False, {}
        else:
            # Invalid action
            self.done = True
            return self.board.copy(), invalid_move_penalty, self.done, False, {}

    def render(self):
        symbols = {0: "_", 1: "A", 2: "B", -1: "#"}
        grid_str = " | ".join([symbols[value] for value in self.board])
        display_str = (
            f"Current player: {'A' if self.current_player == 1 else 'B'}\n"
            f"Grid:\n[ {grid_str} ]"
        )
        print(display_str)
        return display_str

    def valid_moves(self):
        valid_actions = []
        # Check if move forward is valid (action 0)
        player_pos = np.where(self.board == self.current_player)[0]
        if len(player_pos) > 0:
            player_pos = player_pos[0]
            # Determine target cell
            if self.current_player == 1:
                target_pos = player_pos + 1
            else:
                target_pos = player_pos - 1
            # Check if target position is within bounds and empty
            if 0 <= target_pos < len(self.board):
                if self.board[target_pos] == 0:
                    valid_actions.append(0)
        # Check for valid block placements (actions 1 to 5)
        for action in range(1, 6):
            cell_index = action - 1
            opponent_start = 4 if self.current_player == 1 else 0
            own_start = 0 if self.current_player == 1 else 4
            # Skip opponent's and own starting cells
            if cell_index == opponent_start or cell_index == own_start:
                continue
            if self.board[cell_index] == 0:
                valid_actions.append(action)
        return valid_actions
