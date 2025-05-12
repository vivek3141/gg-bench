import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - move 1 point forward, 1 - move 2 points forward
        self.action_space = spaces.Discrete(2)
        # Observation space: positions from 1 to 11 (indices 0 to 10)
        # Values: 1 for current player, -1 for opponent, 0 for empty
        self.observation_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.int8)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board with zeros
        self.board = np.zeros(11, dtype=np.int8)
        # Set starting positions
        self.board[0] = 1  # Player 1 starts at Point 1
        self.board[10] = -1  # Player 2 starts at Point 11
        # Player 1 starts the game
        self.current_player = 1
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), -10, True, False, {}
        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid action
            self.done = True
            return self.board.copy(), -10, True, False, {}

        # Get current positions
        current_pos = np.where(self.board == self.current_player)[0][0]
        opponent_pos = np.where(self.board == -self.current_player)[0][0]

        # Determine move distance
        move_distance = action + 1  # Action 0 -> move 1, Action 1 -> move 2

        # Calculate desired new position based on the current player
        if self.current_player == 1:
            desired_pos = current_pos + move_distance
            # Movement towards center, cannot go beyond Point 6 (index 5)
            desired_pos = min(desired_pos, 5)
            # Check for passing or landing on opponent
            if desired_pos >= opponent_pos:
                # Adjust move if possible
                if move_distance == 2:
                    adjusted_pos = current_pos + 1
                    if adjusted_pos >= opponent_pos:
                        # Cannot move, lose turn
                        pass  # No movement
                    else:
                        desired_pos = adjusted_pos
                else:
                    # Cannot move, lose turn
                    pass  # No movement
            # Move the player if movement is possible
            if desired_pos > current_pos and desired_pos < opponent_pos:
                self.board[current_pos] = 0
                self.board[desired_pos] = self.current_player
        else:  # current_player == -1
            desired_pos = current_pos - move_distance
            # Movement towards center, cannot go beyond Point 6 (index 5)
            desired_pos = max(desired_pos, 5)
            # Check for passing or landing on opponent
            if desired_pos <= opponent_pos:
                # Adjust move if possible
                if move_distance == 2:
                    adjusted_pos = current_pos - 1
                    if adjusted_pos <= opponent_pos:
                        # Cannot move, lose turn
                        pass  # No movement
                    else:
                        desired_pos = adjusted_pos
                else:
                    # Cannot move, lose turn
                    pass  # No movement
            # Move the player if movement is possible
            if desired_pos < current_pos and desired_pos > opponent_pos:
                self.board[current_pos] = 0
                self.board[desired_pos] = self.current_player

        # Check for win condition
        if (self.current_player == 1 and np.where(self.board == 1)[0][0] == 5) or (
            self.current_player == -1 and np.where(self.board == -1)[0][0] == 5
        ):
            # Current player wins
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Check if both players cannot move (unlikely in this game)
        # Switch to the next player
        self.current_player *= -1

        return self.board.copy(), 0, False, False, {}

    def render(self):
        # Create a visual representation of the game state
        board_str = ""
        for idx in range(11):
            if self.board[idx] == 1:
                board_str += "[P1]"
            elif self.board[idx] == -1:
                board_str += "[P2]"
            else:
                board_str += f" {idx + 1} "
            if idx != 10:
                board_str += " - "
        return board_str

    def valid_moves(self):
        # Determine valid actions for the current player
        valid_actions = []
        # Get current positions
        current_pos = np.where(self.board == self.current_player)[0][0]
        opponent_pos = np.where(self.board == -self.current_player)[0][0]

        # Possible move distances
        for action in range(2):  # Actions 0 and 1
            move_distance = action + 1

            if self.current_player == 1:
                desired_pos = current_pos + move_distance
                # Cannot go beyond Point 6
                desired_pos = min(desired_pos, 5)
                # Check for passing or landing on opponent
                if desired_pos >= opponent_pos:
                    continue  # Invalid move
                if desired_pos > current_pos:
                    valid_actions.append(action)
            else:  # current_player == -1
                desired_pos = current_pos - move_distance
                # Cannot go beyond Point 6
                desired_pos = max(desired_pos, 5)
                # Check for passing or landing on opponent
                if desired_pos <= opponent_pos:
                    continue  # Invalid move
                if desired_pos < current_pos:
                    valid_actions.append(action)
        return valid_actions
