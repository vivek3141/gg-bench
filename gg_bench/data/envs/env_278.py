import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)

        # Observation space is a Box space with shape (18,)
        # First 9 elements are cell numbers (1-9)
        # Next 9 elements are markers (-1, 0, 1)
        low = np.array([1] * 9 + [-1] * 9)
        high = np.array([9] * 9 + [1] * 9)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board numbers
        self.cell_numbers = np.random.permutation(np.arange(1, 10))
        # Initialize markers
        self.markers = np.zeros(9, dtype=np.int32)
        # Current player: 1 (Player 1) or -1 (Player 2)
        self.current_player = 1
        self.done = False
        # Prepare observation
        observation = np.concatenate((self.cell_numbers, self.markers))
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            observation = np.concatenate((self.cell_numbers, self.markers))
            return observation, 0, True, False, {}  # Game already over

        # Check if action is valid
        if self.markers[action] != 0:
            # Invalid move
            observation = np.concatenate((self.cell_numbers, self.markers))
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # Reward, Terminated, Truncated, Info

        # Place the marker
        self.markers[action] = self.current_player

        # Check for win
        win = self.check_win()
        if win:
            self.done = True
            observation = np.concatenate((self.cell_numbers, self.markers))
            return observation, 1, True, False, {}  # Current player wins

        # Check for draw
        if np.all(self.markers != 0):
            self.done = True
            observation = np.concatenate((self.cell_numbers, self.markers))
            return observation, 0, True, False, {}  # Draw

        # Switch player
        self.current_player *= -1
        observation = np.concatenate((self.cell_numbers, self.markers))
        return observation, 0, False, False, {}  # Continue game

    def check_win(self):
        win_combinations = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],  # Rows
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],  # Columns
            [0, 4, 8],
            [2, 4, 6],  # Diagonals
        ]
        for combo in win_combinations:
            if np.all(self.markers[combo] == self.current_player):
                # Get the numbers in those cells
                num_seq = self.cell_numbers[combo]
                # Check if they form an arithmetic sequence
                if self.is_arithmetic_sequence(num_seq):
                    return True  # Winning condition met
        return False

    def is_arithmetic_sequence(self, seq):
        # Check if the sequence has a common difference
        diff1 = seq[1] - seq[0]
        diff2 = seq[2] - seq[1]
        return diff1 == diff2

    def render(self):
        board_str = "Current Grid:\n"
        for i in range(3):
            board_str += "|"
            for j in range(3):
                idx = i * 3 + j
                if self.markers[idx] == 1:
                    board_str += " X |"
                elif self.markers[idx] == -1:
                    board_str += " O |"
                else:
                    board_str += f" {self.cell_numbers[idx]} |"
            board_str += "\n"
        print(board_str)

    def valid_moves(self):
        return [i for i in range(9) if self.markers[i] == 0]
