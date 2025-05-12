import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to territories 0 to 6
        self.action_space = spaces.Discrete(7)
        # Observation is the state of the territories: -1, 0, or 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(7, dtype=np.int8)  # Territories are unclaimed
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {"error": "Game is over. Please reset the environment."},
            )

        if action < 0 or action >= 7 or self.board[action] != 0:
            # Invalid move
            self.done = True
            return (
                self.board.copy(),
                -10,
                True,
                False,
                {
                    "error": "Invalid action. Territory already claimed or out of bounds."
                },
            )

        # Valid move, claim the territory
        self.board[action] = self.current_player

        # Check for immediate victory
        player_indices = np.where(self.board == self.current_player)[0]
        if self.check_victory(player_indices):
            self.done = True
            return self.board.copy(), 1, True, False, {}

        # Check if all territories are claimed
        if np.all(self.board != 0):
            # All territories are claimed, determine the winner
            winner = self.determine_winner()
            self.done = True
            if winner == self.current_player:
                return self.board.copy(), 1, True, False, {}
            elif winner == 0:
                return self.board.copy(), 0, True, False, {}
            else:
                return self.board.copy(), -1, True, False, {}

        # Game continues, switch to the other player
        self.current_player *= -1
        return self.board.copy(), 0, False, False, {}

    def render(self):
        board_str = "Territories:\n"
        for i in range(7):
            state = self.board[i]
            if state == 1:
                owner = "P1"
            elif state == -1:
                owner = "P2"
            else:
                owner = "Unclaimed"
            board_str += f" {i + 1}: {owner}\n"
        return board_str

    def valid_moves(self):
        return [i for i in range(7) if self.board[i] == 0]

    def check_victory(self, indices):
        if len(indices) < 3:
            return False
        sorted_indices = np.sort(indices)
        for i in range(len(sorted_indices) - 2):
            if (
                sorted_indices[i + 1] == sorted_indices[i] + 1
                and sorted_indices[i + 2] == sorted_indices[i] + 2
            ):
                return True
        return False

    def get_longest_chain(self, indices):
        if len(indices) == 0:
            return 0
        sorted_indices = np.sort(indices)
        longest = 1
        current = 1
        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] == sorted_indices[i - 1] + 1:
                current += 1
                if current > longest:
                    longest = current
            else:
                current = 1
        return longest

    def determine_winner(self):
        # Calculate longest chains for both players
        p1_indices = np.where(self.board == 1)[0]
        p2_indices = np.where(self.board == -1)[0]
        longest_chain_p1 = self.get_longest_chain(p1_indices)
        longest_chain_p2 = self.get_longest_chain(p2_indices)

        if longest_chain_p1 > longest_chain_p2:
            return 1  # Player 1 wins
        elif longest_chain_p2 > longest_chain_p1:
            return -1  # Player 2 wins
        else:
            # Longest chains are equal, check total territories claimed
            num_p1 = len(p1_indices)
            num_p2 = len(p2_indices)
            if num_p1 > num_p2:
                return 1  # Player 1 wins
            elif num_p2 > num_p1:
                return -1  # Player 2 wins
            else:
                return 0  # Tie
