import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space:
        # Action 0: Move forward
        # Actions 1-9: Place obstacle at position (action)
        self.action_space = spaces.Discrete(10)

        # Define observation space:
        # -1: Obstacle
        # 0: Empty
        # 1: Player 1's piece
        # 2: Player 2's piece
        self.observation_space = spaces.Box(low=-1, high=2, shape=(9,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board
        self.board = np.zeros(9, dtype=np.int8)
        # Place players at starting positions
        self.board[0] = 1  # Player 1 at position 1 (index 0)
        self.board[8] = 2  # Player 2 at position 9 (index 8)
        # Set current player (1 or 2)
        self.current_player = 1  # Player 1 starts first
        # Record positions of players
        self.player_positions = {1: 0, 2: 8}
        # Game over flag
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        reward = 0
        info = {}
        invalid_move = False

        if action == 0:
            # Move forward
            position = self.player_positions[self.current_player]
            target_position = None
            if self.current_player == 1:
                target_position = position + 1
            elif self.current_player == 2:
                target_position = position - 1
            else:
                # Invalid player number
                invalid_move = True

            if target_position < 0 or target_position > 8:
                # Target position is off the board
                invalid_move = True
            else:
                target_value = self.board[target_position]
                if target_value == 0:
                    # Empty position, move the piece
                    self.board[position] = 0
                    self.board[target_position] = self.current_player
                    self.player_positions[self.current_player] = target_position
                    # Check for win
                    if (self.current_player == 1 and target_position == 8) or (
                        self.current_player == 2 and target_position == 0
                    ):
                        # Player wins
                        reward = 1
                        self.done = True
                elif target_value == -1:
                    # Obstacle encountered, return to starting position
                    self.board[position] = 0
                    starting_position = 0 if self.current_player == 1 else 8
                    self.board[starting_position] = self.current_player
                    self.player_positions[self.current_player] = starting_position
                else:
                    # Cannot move onto another player's piece
                    invalid_move = True
        elif 1 <= action <= 9:
            # Place obstacle at position (action - 1)
            obstacle_position = action - 1
            position_value = self.board[obstacle_position]
            if position_value == 0:
                # Place obstacle
                self.board[obstacle_position] = -1
            else:
                # Cannot place obstacle on non-empty position
                invalid_move = True
        else:
            # Invalid action
            invalid_move = True

        if invalid_move:
            reward = -10
            self.done = True
            info["invalid_move"] = True
            return self.board.copy(), reward, self.done, False, info

        # Switch players
        self.current_player = 2 if self.current_player == 1 else 1

        return self.board.copy(), reward, self.done, False, info

    def render(self):
        # Return a string representation of the board
        board_str = ""
        for idx in range(9):
            val = self.board[idx]
            if val == 0:
                board_str += " _ "  # Empty
            elif val == -1:
                board_str += " X "  # Obstacle
            elif val == 1:
                board_str += "P1 "  # Player 1
            elif val == 2:
                board_str += "P2 "  # Player 2
            if idx != 8:
                board_str += "|"
        return board_str

    def valid_moves(self):
        valid_actions = []
        if self.done:
            return valid_actions

        # Check if move forward is valid
        position = self.player_positions[self.current_player]
        if self.current_player == 1:
            target_position = position + 1
        else:
            target_position = position - 1

        if 0 <= target_position <= 8:
            target_value = self.board[target_position]
            if target_value != -1 and target_value != 1 and target_value != 2:
                # Can attempt to move forward
                valid_actions.append(0)

        # For placing obstacles, actions 1-9
        for action in range(1, 10):
            obstacle_position = action - 1
            if self.board[obstacle_position] == 0:
                # Empty position, can place obstacle
                valid_actions.append(action)
        return valid_actions
