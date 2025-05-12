import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 (move 1 position forward), 1 (move 2 positions forward)
        self.action_space = spaces.Discrete(2)

        # Observations: Array of 11 positions (0 to 10), values -1, 0, 1
        # -1: Player 2's unit, 0: empty, 1: Player 1's unit
        self.observation_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Initialize the board positions
        self.board = np.zeros(11, dtype=np.int8)
        self.positions = {1: 0, -1: 10}  # Player units' positions

        # Place the units on their bases
        self.board[0] = 1  # Player 1's unit
        self.board[10] = -1  # Player 2's unit

        return self.board.copy(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        move = action + 1  # Map action 0->1, action 1->2

        current_pos = self.positions[self.current_player]

        # Calculate new position based on the current player
        if self.current_player == 1:
            new_pos = current_pos + move
        else:
            new_pos = current_pos - move

        # Check for invalid move (out of bounds)
        if new_pos < 0 or new_pos > 10:
            self.done = True
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Move the unit
        self.board[current_pos] = 0  # Clear old position

        # Battle check
        if self.board[new_pos] == -self.current_player:
            # Battle occurs; attacking player wins
            defender = -self.current_player
            self.board[new_pos] = 0  # Remove defender's unit from the board
            # Send defender back to their base
            base_pos = 0 if defender == 1 else 10
            self.positions[defender] = base_pos
            self.board[base_pos] = defender

        # Place current player's unit on the new position
        self.board[new_pos] = self.current_player
        self.positions[self.current_player] = new_pos

        # Check for victory
        if (self.current_player == 1 and new_pos == 10) or (
            self.current_player == -1 and new_pos == 0
        ):
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Switch player
        self.current_player *= -1
        return self.board.copy(), 0, False, False, {}

    def render(self):
        # Generate a string representation of the board
        board_str = "Positions:"
        for i in range(11):
            if self.board[i] == 1:
                board_str += f" [{i} P1]"
            elif self.board[i] == -1:
                board_str += f" [{i} P2]"
            else:
                board_str += f" [{i}   ]"
        return board_str

    def valid_moves(self):
        # Determine valid moves for the current player
        valid_actions = []
        for action in range(2):
            move = action + 1
            current_pos = self.positions[self.current_player]

            if self.current_player == 1:
                new_pos = current_pos + move
            else:
                new_pos = current_pos - move

            if 0 <= new_pos <= 10:
                valid_actions.append(action)
        return valid_actions
