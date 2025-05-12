import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is 16 discrete actions (cells 0 to 15)
        self.action_space = spaces.Discrete(16)

        # The observation space is a vector of 17 integers:
        # - The first 16 elements represent the board state
        #   - 0: unclaimed and unblocked cell
        #   - 1: claimed by the current player
        #   - -1: claimed by the opponent
        #   - -2: blocked cell
        # - The 17th element represents the current player (1 or -1)
        self.observation_space = spaces.Box(low=-2, high=1, shape=(17,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board: all cells are unclaimed and unblocked (0)
        self.board = np.zeros(16, dtype=np.int8)
        # Player 1 starts first (represented by 1), Player 2 is represented by -1
        self.current_player = 1
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Return the observation: board state concatenated with the current player
        obs = np.append(self.board, self.current_player)
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if the action is valid
        if action < 0 or action >= 16 or self.board[action] != 0:
            # Invalid action
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Claim the selected cell
        self.board[action] = self.current_player

        # Block adjacent cells
        adjacent_indices = self.get_adjacent_indices(action)
        for idx in adjacent_indices:
            if self.board[idx] == 0:
                self.board[idx] = -2  # Mark as blocked

        # Switch to the opponent
        self.current_player *= -1

        # Check if the opponent has any valid moves
        if not self.valid_moves():
            # The player who just moved wins
            self.done = True
            reward = 1  # Reward for winning
            # Switch back to the winning player for observation
            self.current_player *= -1
            return self._get_obs(), reward, True, False, {}

        # Game continues
        return self._get_obs(), 0, False, False, {}

    def valid_moves(self):
        # Return a list of indices of valid moves (unclaimed and unblocked cells)
        return [i for i in range(16) if self.board[i] == 0]

    def render(self):
        # Create a visual representation of the board state
        symbols = {0: " ", 1: "X", -1: "O", -2: "-"}
        board_symbols = [symbols.get(self.board[i], "?") for i in range(16)]
        rows = []
        for i in range(0, 16, 4):
            row = " | ".join(board_symbols[i : i + 4])
            rows.append(row)
        board_str = "\n---------\n".join(rows)
        player_str = f"Current Player: {'X' if self.current_player == 1 else 'O'}"
        return f"{player_str}\n{board_str}"

    def get_adjacent_indices(self, idx):
        # Calculate the adjacent cell indices for a given cell index
        adjacent_indices = []
        row = idx // 4
        col = idx % 4
        for r in range(max(0, row - 1), min(4, row + 2)):
            for c in range(max(0, col - 1), min(4, col + 2)):
                adj_idx = r * 4 + c
                if adj_idx != idx:
                    adjacent_indices.append(adj_idx)
        return adjacent_indices
