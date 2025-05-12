import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 20 positions corresponding to numbers 1 to 20
        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.int8)

        # Initialize the board and game variables
        self.board = np.zeros(
            20, dtype=np.int8
        )  # 0: unoccupied, 1: Player 1 (X), -1: Player 2 (O)
        self.current_player = 1  # Starts with Player 1
        self.done = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(20, dtype=np.int8)
        self.current_player = 1
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        # Check if the action is valid
        if self.done or action < 0 or action >= 20 or self.board[action] != 0:
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {},
            )  # Invalid move: reward -10, game over

        # Place the player's marker on the chosen number
        self.board[action] = self.current_player

        # Check for victory condition
        player_positions = np.where(self.board == self.current_player)[0]
        if len(player_positions) >= 3:
            # Convert positions to numbers (1-based indexing)
            player_numbers = player_positions + 1
            num_numbers = len(player_numbers)
            found = False
            # Check all pairs for divisor relationship
            for i in range(num_numbers):
                for j in range(num_numbers):
                    if i != j:
                        a = player_numbers[i]
                        b = player_numbers[j]
                        if b % a == 0:
                            found = True
                            break
                if found:
                    break
            if found:
                self.done = True
                return (
                    self.board.copy(),
                    1,
                    True,
                    False,
                    {},
                )  # Current player wins: reward 1

        # Switch to the other player
        self.current_player *= -1  # Change player: 1 becomes -1, and -1 becomes 1
        return (
            self.board.copy(),
            0,
            False,
            False,
            {},
        )  # Valid move: reward 0, game continues

    def render(self):
        # Create a visual representation of the board
        board_str = ""
        for i in range(20):
            if self.board[i] == 1:
                marker = "[X]"
            elif self.board[i] == -1:
                marker = "[O]"
            else:
                marker = f"{i+1:2d} "  # Unoccupied numbers
            board_str += marker + " "
        return board_str.strip()

    def valid_moves(self):
        # Return a list of unoccupied positions
        return [i for i in range(20) if self.board[i] == 0]
