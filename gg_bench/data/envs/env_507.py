import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(25,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(25, dtype=np.float32)
        self.current_player = 1  # Player 1 starts (uses 1), Player 2 uses -1
        self.done = False
        self.player1_claims = 0
        self.player2_claims = 0
        self.chain_claim_in_progress = False
        self.last_action = None
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {},
            )  # Cannot play if game is over

        if action < 0 or action >= 25:
            self.done = True
            return self.board.copy(), -10, True, False, {}  # Invalid action

        if self.board[action] != 0:
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {},
            )  # Invalid move, cell already claimed

        # Claim the cell
        self.board[action] = self.current_player

        # Update claimed cells count
        if self.current_player == 1:
            self.player1_claims += 1
        else:
            self.player2_claims += 1

        # Check for victory
        if self.player1_claims >= 13 or self.player2_claims >= 13:
            self.done = True
            reward = 1  # Current player wins
            return self.board.copy(), reward, True, False, {}

        # Chain claim logic
        if not self.chain_claim_in_progress:
            # First move in the player's turn
            self.last_action = action  # Store the last action
            # Check adjacency with own claimed cells (excluding the new one)
            adjacent_indices = self.get_adjacent_indices(action)
            has_adjacent_owned_cell = False
            for idx in adjacent_indices:
                if self.board[idx] == self.current_player:
                    has_adjacent_owned_cell = True
                    break
            if has_adjacent_owned_cell:
                self.chain_claim_in_progress = True  # Eligible for chain claim
                reward = 0
                return (
                    self.board.copy(),
                    reward,
                    False,
                    False,
                    {},
                )  # Do not switch player yet
            else:
                # No chain claim, switch player
                self.current_player *= -1
                reward = 0
                return self.board.copy(), reward, False, False, {}
        else:
            # Chain claim move
            adjacent_indices = self.get_adjacent_indices(self.last_action)
            if action not in adjacent_indices or self.board[action] != 0:
                self.done = True
                return self.board.copy(), -10, True, False, {}  # Invalid chain claim

            # Claim the cell
            self.board[action] = self.current_player

            # Update claimed cells count
            if self.current_player == 1:
                self.player1_claims += 1
            else:
                self.player2_claims += 1

            # Check for victory
            if self.player1_claims >= 13 or self.player2_claims >= 13:
                self.done = True
                reward = 1  # Current player wins
                return self.board.copy(), reward, True, False, {}

            # Chain claim done, switch player
            self.chain_claim_in_progress = False
            self.last_action = None
            self.current_player *= -1
            reward = 0
            return self.board.copy(), reward, False, False, {}

    def get_adjacent_indices(self, index):
        row = index // 5
        col = index % 5
        adjacent = []
        # Up
        if row > 0:
            adjacent.append((row - 1) * 5 + col)
        # Down
        if row < 4:
            adjacent.append((row + 1) * 5 + col)
        # Left
        if col > 0:
            adjacent.append(row * 5 + (col - 1))
        # Right
        if col < 4:
            adjacent.append(row * 5 + (col + 1))
        return adjacent

    def render(self):
        board_str = "\nCurrent board state:\n"
        symbols = {1: " X ", -1: " O ", 0: " . "}
        for row in range(5):
            board_str += "|"
            for col in range(5):
                index = row * 5 + col
                board_str += symbols[self.board[index]]
            board_str += "|\n"
        return board_str

    def valid_moves(self):
        if self.done:
            return []
        valid_actions = []
        if self.chain_claim_in_progress:
            # Chain claim, valid moves are unclaimed adjacent cells to last_action
            adjacent_indices = self.get_adjacent_indices(self.last_action)
            for idx in adjacent_indices:
                if self.board[idx] == 0:
                    valid_actions.append(idx)
        else:
            # Regular move, any unclaimed cell
            valid_actions = [idx for idx in range(25) if self.board[idx] == 0]
        return valid_actions
