import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - move 1 position, 1 - move 2 positions
        self.action_space = spaces.Discrete(2)

        # Observation space: board positions from positions 1 to 7 (indices 0 to 6)
        # Values can be -1 (P2), 0 (empty), 1 (P1)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board
        self.board = np.zeros(7, dtype=np.int8)
        # Place players' tokens
        self.board[0] = 1  # P1 at position 1
        self.board[6] = -1  # P2 at position 7
        # Initialize positions
        self.p1_position = 0
        self.p2_position = 6
        # Player 1 starts first
        self.current_player = 1  # 1 for P1, -1 for P2
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        reward = 0

        # Determine current player's position
        if self.current_player == 1:
            current_position = self.p1_position
        else:
            current_position = self.p2_position

        # Map action to movement (action 0: move 1, action 1: move 2)
        move_distance = action + 1

        # Calculate new position
        if self.current_player == 1:
            new_position = current_position + move_distance
            # Check if new_position goes beyond the flag position
            if new_position > 3:
                # Cannot move past the flag
                self.done = True
                return self.board.copy(), -10, True, False, {}
        else:
            new_position = current_position - move_distance
            # Check if new_position goes beyond the flag position
            if new_position < 3:
                # Cannot move past the flag
                self.done = True
                return self.board.copy(), -10, True, False, {}

        # Check for collision (cannot land on the other player's position unless it's the flag position)
        if new_position != 3 and self.board[new_position] != 0:
            # Invalid move, collision detected
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Update the board
        self.board[current_position] = 0  # Remove player's token from current position
        self.board[new_position] = (
            self.current_player
        )  # Place player's token at new position

        # Update player's position
        if self.current_player == 1:
            self.p1_position = new_position
        else:
            self.p2_position = new_position

        # Check for win
        if new_position == 3:
            # Player captures the flag and wins
            self.done = True
            reward = 1
            return self.board.copy(), reward, True, False, {}

        # Switch to the other player
        self.current_player *= -1

        return self.board.copy(), reward, False, False, {}

    def render(self):
        # Visual representation of the board
        board_str = ""
        for i in range(7):
            pos = self.board[i]
            if i == 3:
                # Position 4, the flag
                if pos == 1:
                    board_str += "[P1F]"
                elif pos == -1:
                    board_str += "[P2F]"
                else:
                    board_str += "[ F ]"
            else:
                if pos == 1:
                    board_str += "[P1 ]"
                elif pos == -1:
                    board_str += "[P2 ]"
                else:
                    board_str += "[    ]"
        return board_str

    def valid_moves(self):
        # Returns list of valid actions (0 or 1) for the current player
        valid_actions = []
        for action in [0, 1]:
            move_distance = action + 1
            if self.current_player == 1:
                new_position = self.p1_position + move_distance
                if new_position > 3:
                    continue  # Cannot move past the flag
            else:
                new_position = self.p2_position - move_distance
                if new_position < 3:
                    continue  # Cannot move past the flag

            # Check for collision
            if new_position != 3 and self.board[new_position] != 0:
                continue  # Invalid move

            valid_actions.append(action)
        return valid_actions
