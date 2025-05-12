import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.int8)

        # Initialize the board
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(11, dtype=np.int8)
        self.current_player = 1  # Player 1 is 1 (A), Player 2 is -1 (B)
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}

        valid_moves = self.valid_moves()
        if action not in valid_moves:
            # Invalid move
            self.done = True
            reward = -10
            return self.board.copy(), reward, True, False, {}

        # Apply the action
        self.board[action] = self.current_player

        # Now check if opponent has any valid moves
        opponent = -self.current_player
        # Temporarily set current_player to opponent for valid_moves calculation
        self.current_player = opponent
        opponent_valid_moves = self.valid_moves()
        if len(opponent_valid_moves) == 0:
            # Opponent has no valid moves, current player wins
            self.done = True
            # Switch back to winner for clarity
            self.current_player = -self.current_player
            reward = 1
            return self.board.copy(), reward, True, False, {}
        else:
            # Game continues
            # self.current_player remains set to opponent for next turn
            reward = 0
            return self.board.copy(), reward, False, False, {}

    def valid_moves(self):
        if self.done:
            return []

        board = self.board
        player = self.current_player

        player_positions = np.where(board == player)[0]

        if len(player_positions) == 0:
            # Player has not yet claimed any nodes
            # Can claim position 0 or 10 (positions 1 or 11)
            valid_positions = []
            if board[0] == 0:
                valid_positions.append(0)
            if board[10] == 0:
                valid_positions.append(10)
            return valid_positions

        else:
            # Player can claim unclaimed nodes adjacent to their claimed nodes
            valid_positions = set()
            for pos in player_positions:
                # Check left neighbor
                if pos - 1 >= 0 and board[pos - 1] == 0:
                    valid_positions.add(pos - 1)
                # Check right neighbor
                if pos + 1 <= 10 and board[pos + 1] == 0:
                    valid_positions.add(pos + 1)
            return list(valid_positions)

    def render(self):
        symbols = {1: "A", -1: "B", 0: "."}
        grid_str = " ".join([symbols[self.board[i]] for i in range(11)])
        print(f"Current player: {'A' if self.current_player == 1 else 'B'}")
        print(f"Grid: {grid_str}")
