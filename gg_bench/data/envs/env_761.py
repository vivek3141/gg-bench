import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4, 4), dtype=np.int8)

        # Winning and losing conditions
        self.winning_positions = []
        self.losing_positions = []

        # Define losing positions (lines of four)
        # Rows
        for r in range(4):
            self.losing_positions.append([r * 4 + c for c in range(4)])
        # Columns
        for c in range(4):
            self.losing_positions.append([r * 4 + c for r in range(4)])
        # Diagonals
        self.losing_positions.append([0, 5, 10, 15])  # Main diagonal
        self.losing_positions.append([3, 6, 9, 12])  # Anti-diagonal

        # Define winning positions (lines of exactly three contiguous cells)
        # Rows
        for r in range(4):
            self.winning_positions.append([r * 4 + c for c in [0, 1, 2]])
            self.winning_positions.append([r * 4 + c for c in [1, 2, 3]])
        # Columns
        for c in range(4):
            self.winning_positions.append([r * 4 + c for r in [0, 1, 2]])
            self.winning_positions.append([r * 4 + c for r in [1, 2, 3]])
        # Diagonals
        self.winning_positions.append([0, 5, 10])
        self.winning_positions.append([5, 10, 15])
        self.winning_positions.append([3, 6, 9])
        self.winning_positions.append([6, 9, 12])

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(16, dtype=np.int8)
        self.current_player = 1
        self.done = False
        return self.board.reshape(4, 4), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                self.board.reshape(4, 4),
                0,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        if not self.action_space.contains(action):
            return (
                self.board.reshape(4, 4),
                -10,
                True,
                False,
                {"invalid_action": "Action out of bounds"},
            )

        if self.board[action] != 0:
            return (
                self.board.reshape(4, 4),
                -10,
                True,
                False,
                {"invalid_action": "Cell is already occupied"},
            )

        self.board[action] = self.current_player

        reward = -10  # Penalty for any valid move

        # Check for lose condition first
        for positions in self.losing_positions:
            if all(self.board[pos] == self.current_player for pos in positions):
                self.done = True
                # reward remains -10
                return (
                    self.board.reshape(4, 4),
                    reward,
                    True,
                    False,
                    {"result": "loss"},
                )

        # Check for win condition
        for positions in self.winning_positions:
            if all(self.board[pos] == self.current_player for pos in positions):
                self.done = True
                reward = 1  # Override penalty reward with win reward
                return (
                    self.board.reshape(4, 4),
                    reward,
                    True,
                    False,
                    {"result": "win"},
                )

        # No win or loss, continue game
        # reward remains -10

        # Switch player
        self.current_player *= -1

        return self.board.reshape(4, 4), reward, False, False, {}

    def render(self):
        board_str = "    1   2   3   4\n"
        board_str += "  +---+---+---+---+\n"
        for i in range(4):
            row_str = f"{i+1} |"
            for j in range(4):
                pos = i * 4 + j
                if self.board[pos] == 1:
                    row_str += " X |"
                elif self.board[pos] == -1:
                    row_str += " O |"
                else:
                    row_str += "   |"
            board_str += row_str + "\n"
            board_str += "  +---+---+---+---+\n"
        return board_str

    def valid_moves(self):
        return [i for i in range(16) if self.board[i] == 0]
