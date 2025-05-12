import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.int8)

        # Initialize the number line and game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_line = np.zeros(10, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.number_line.copy(), {}  # Return observation and info

    def step(self, action):
        info = {}
        if self.done:
            return self.number_line.copy(), 0, True, False, info

        # Validate action
        if self.number_line[action] != 0:
            self.done = True
            return (
                self.number_line.copy(),
                -10,
                True,
                False,
                info,
            )  # Observation, reward, terminated, truncated, info

        # Claim the number
        self.number_line[action] = self.current_player

        # Check for win condition
        player_positions = np.where(self.number_line == self.current_player)[0]
        player_numbers = player_positions + 1  # Convert indices to numbers (1-10)

        # Check for four consecutive numbers
        if len(player_numbers) >= 4:
            sorted_numbers = np.sort(player_numbers)
            consecutive_counts = 1
            for i in range(1, len(sorted_numbers)):
                if sorted_numbers[i] == sorted_numbers[i - 1] + 1:
                    consecutive_counts += 1
                    if consecutive_counts >= 4:
                        self.done = True
                        return (
                            self.number_line.copy(),
                            1,
                            True,
                            False,
                            info,
                        )  # Current player wins
                else:
                    consecutive_counts = 1

        # Check for a draw and tiebreaker if all numbers are claimed
        if np.all(self.number_line != 0):
            self.done = True
            # Perform tiebreaker
            winner = self.tiebreaker()
            if winner == self.current_player:
                return self.number_line.copy(), 1, True, False, info
            else:
                return self.number_line.copy(), -1, True, False, info

        # Switch player
        self.current_player *= -1
        return self.number_line.copy(), 0, False, False, info

    def tiebreaker(self):
        # Find the longest consecutive sequence for each player
        def longest_consecutive(player_numbers):
            if len(player_numbers) == 0:
                return 0, None
            sorted_numbers = np.sort(player_numbers)
            max_length = 1
            current_length = 1
            start_number = sorted_numbers[0]
            longest_start = sorted_numbers[0]

            for i in range(1, len(sorted_numbers)):
                if sorted_numbers[i] == sorted_numbers[i - 1] + 1:
                    current_length += 1
                    if current_length > max_length:
                        max_length = current_length
                        longest_start = sorted_numbers[i - current_length + 1]
                else:
                    current_length = 1

            return max_length, longest_start

        player1_positions = np.where(self.number_line == 1)[0]
        player1_numbers = player1_positions + 1
        player2_positions = np.where(self.number_line == -1)[0]
        player2_numbers = player2_positions + 1

        p1_length, p1_start = longest_consecutive(player1_numbers)
        p2_length, p2_start = longest_consecutive(player2_numbers)

        if p1_length > p2_length:
            return 1
        elif p2_length > p1_length:
            return -1
        else:
            # Lengths are equal, compare starting numbers
            if p1_start < p2_start:
                return 1
            elif p2_start < p1_start:
                return -1
            else:
                return 0  # It's a tie (unlikely in this game)

    def render(self):
        number_line_str = ""
        for i in range(10):
            marker = ""
            if self.number_line[i] == 1:
                marker = "[X]"
            elif self.number_line[i] == -1:
                marker = "[O]"
            else:
                marker = f"{i + 1}"
            number_line_str += f"{marker} "
        return number_line_str.strip()

    def valid_moves(self):
        return [i for i in range(10) if self.number_line[i] == 0]
