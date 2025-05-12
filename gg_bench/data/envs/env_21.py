import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to positions 1 to 7 (indices 0 to 6)
        self.action_space = spaces.Discrete(7)

        # Observation space: positions 1 to 7 (indices 0 to 6)
        # Each position can be:
        #  0: empty
        #  1: Player 1's Key
        #  2: Player 1's Lock
        # -1: Player 2's Key
        # -2: Player 2's Lock
        self.observation_space = spaces.Box(low=-2, high=2, shape=(7,), dtype=np.int8)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initial board setup
        # Positions are indices 0 to 6 corresponding to positions 1 to 7
        self.board = np.array([1, 2, 2, 0, -2, -2, -1], dtype=np.int8)

        # Token positions for each player
        # Player 1: positions of Key and Locks
        self.p1_key_pos = 0
        self.p1_locks_pos = [1, 2]

        # Player 2: positions of Key and Locks
        self.p2_key_pos = 6
        self.p2_locks_pos = [5, 4]

        # Current player: 1 for Player 1, -1 for Player 2
        self.current_player = 1

        # Game over flag
        self.done = False

        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        reward = 0

        # Check if action is valid
        # For the current player, attempt to move the token at position action
        pos = action  # Position index (0 to 6)

        # Check if there is a token of the current player at pos
        token = self.board[pos]
        if self.current_player == 1:
            player_key = 1
            player_lock = 2
            opponent_key = -1
            opponent_lock = -2
            direction = 1  # Move right
            next_pos = pos + 1
        else:
            player_key = -1
            player_lock = -2
            opponent_key = 1
            opponent_lock = 2
            direction = -1  # Move left
            next_pos = pos - 1

        if token != player_key and token != player_lock:
            # Invalid move: no player's token at the position
            self.done = True
            reward = -10
            return self.board.copy(), reward, True, False, {}

        # Check if next position is within bounds
        if next_pos < 0 or next_pos > 6:
            # Invalid move: cannot move off the board
            self.done = True
            reward = -10
            return self.board.copy(), reward, True, False, {}

        # Check if next position is occupied by own token
        next_token = self.board[next_pos]
        if next_token == player_key or next_token == player_lock:
            # Invalid move: cannot move into a position occupied by own token
            self.done = True
            reward = -10
            return self.board.copy(), reward, True, False, {}

        # Move the token
        self.board[pos] = 0  # Clear current position

        capture = False

        # Handle capturing
        if next_token == opponent_key or next_token == opponent_lock:
            # Determine capture rules
            if token == player_key:
                # Player's Key: can capture opponent's Key or Lock
                capture = True
            elif token == player_lock:
                # Player's Lock
                if next_token == opponent_lock:
                    # Locks can capture opponent's Locks
                    capture = True
                else:
                    # Locks cannot capture opponent's Key
                    capture = False

            if capture:
                # Remove opponent's token
                if next_token == opponent_key:
                    # Captured opponent's Key: win the game
                    self.board[next_pos] = 0
                    self.done = True
                    reward = 1
                    self.board[next_pos] = token  # Move token into captured position
                    return self.board.copy(), reward, True, False, {}
                else:
                    # Captured opponent's Lock
                    self.board[next_pos] = 0
        # Move token into next position
        self.board[next_pos] = token

        # Update token positions
        if self.current_player == 1:
            if token == player_key:
                self.p1_key_pos = next_pos
            else:
                # Update Locks' positions
                if pos in self.p1_locks_pos:
                    self.p1_locks_pos.remove(pos)
                self.p1_locks_pos.append(next_pos)
        else:
            if token == player_key:
                self.p2_key_pos = next_pos
            else:
                # Update Locks' positions
                if pos in self.p2_locks_pos:
                    self.p2_locks_pos.remove(pos)
                self.p2_locks_pos.append(next_pos)

        # Check if moved into opponent's Key without capturing
        if next_token == opponent_key and not capture:
            # Cannot capture opponent's Key with a Lock
            # Reverse the move and penalize
            self.board[pos] = token  # Undo move
            self.board[next_pos] = next_token  # Restore opponent's Key
            self.done = True
            reward = -10
            return self.board.copy(), reward, True, False, {}

        # Check for win condition (if opponent's Key is already captured)
        if self.current_player == 1 and self.board[self.p2_key_pos] == 0:
            # Player 1 has already won
            self.done = True
            reward = 1
            return self.board.copy(), reward, True, False, {}
        elif self.current_player == -1 and self.board[self.p1_key_pos] == 0:
            # Player 2 has already won
            self.done = True
            reward = 1
            return self.board.copy(), reward, True, False, {}

        # Switch current player
        self.current_player *= -1

        # Game continues
        return self.board.copy(), reward, False, False, {}

    def render(self):
        board_str = "Positions: [1][2][3][4][5][6][7]\n"
        tokens_str = ""
        for i in range(7):
            token = self.board[i]
            if token == 0:
                tokens_str += "[  ]"
            elif token == 1:
                tokens_str += "[K1]"
            elif token == 2:
                tokens_str += "[L1]"
            elif token == -1:
                tokens_str += "[K2]"
            elif token == -2:
                tokens_str += "[L2]"
        board_str += tokens_str + "\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        for pos in range(7):
            token = self.board[pos]
            if self.current_player == 1:
                player_key = 1
                player_lock = 2
                direction = 1
                next_pos = pos + 1
            else:
                player_key = -1
                player_lock = -2
                direction = -1
                next_pos = pos - 1

            if token == player_key or token == player_lock:
                # Check if next position is within bounds
                if next_pos < 0 or next_pos > 6:
                    continue
                next_token = self.board[next_pos]
                # Check if next position is not occupied by own token
                if next_token == player_key or next_token == player_lock:
                    continue
                valid_actions.append(pos)
        return valid_actions
