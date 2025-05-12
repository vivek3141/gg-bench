import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 -> move 1 step, 1 -> move 2 steps, 2 -> move 3 steps
        self.action_space = spaces.Discrete(3)

        # Define observation space: a vector representing the 11 tiles on the path
        # -1 represents Player 2, 0 represents an empty tile, 1 represents Player 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board: Player 1 at tile 0, Player 2 at tile 10
        self.board = np.zeros(11, dtype=np.int32)
        self.board[0] = 1  # Player 1
        self.board[10] = -1  # Player 2
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        return self.board, {}  # Return initial observation and info

    def step(self, action):
        if self.done:
            return self.board, 0, self.done, False, {}  # Game is over

        # Map action to number of steps (action: 0 -> 1 step, 1 -> 2 steps, 2 -> 3 steps)
        steps = action + 1

        # Get the current player's position
        position = np.where(self.board == self.current_player)[0][0]

        # Determine the new position based on the current player
        if self.current_player == 1:
            new_position = position + steps
            # Check for valid move (cannot overshoot the flag at tile 5)
            if new_position > 5:
                self.done = True
                return (
                    self.board,
                    -10,
                    self.done,
                    False,
                    {},
                )  # Invalid move, lose the game
        else:
            new_position = position - steps
            # Check for valid move (cannot overshoot the flag at tile 5)
            if new_position < 5:
                self.done = True
                return (
                    self.board,
                    -10,
                    self.done,
                    False,
                    {},
                )  # Invalid move, lose the game

        # Move is valid; update the board
        self.board[position] = 0  # Remove player from current position
        self.board[new_position] = self.current_player  # Place player at new position

        # Check for a win condition
        if new_position == 5:
            self.done = True
            return self.board, 1, self.done, False, {}  # Current player wins

        # Switch to the other player
        self.current_player *= -1
        return self.board, 0, self.done, False, {}  # Continue the game

    def render(self):
        # Create a visual representation of the board state
        board_visual = ""
        for i in range(11):
            if i == 0 and self.board[i] == 1:
                board_visual += "[P1]"
            elif i == 10 and self.board[i] == -1:
                board_visual += "[P2]"
            elif i == 5:
                if self.board[i] == 1:
                    board_visual += "[P1/Flag]"
                elif self.board[i] == -1:
                    board_visual += "[P2/Flag]"
                else:
                    board_visual += "[Flag]"
            else:
                if self.board[i] == 1:
                    board_visual += " P1 "
                elif self.board[i] == -1:
                    board_visual += " P2 "
                else:
                    board_visual += " . "
            if i < 10:
                board_visual += "-"
        return board_visual

    def valid_moves(self):
        # Determine the list of valid moves for the current player
        position = np.where(self.board == self.current_player)[0][0]
        valid_actions = []
        for action in range(
            3
        ):  # Possible actions: 0 (1 step), 1 (2 steps), 2 (3 steps)
            steps = action + 1
            if self.current_player == 1:
                new_position = position + steps
                if new_position <= 5:
                    valid_actions.append(action)
            else:
                new_position = position - steps
                if new_position >= 5:
                    valid_actions.append(action)
        return valid_actions
