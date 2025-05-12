import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 21 possible actions (numbers from 1 to 21)
        self.action_space = spaces.Discrete(21)
        # Observation space is the state of the number line
        # Values: -1 (opponent claimed), 0 (unclaimed), 1 (current player claimed)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(21,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(21, dtype=np.int8)  # Number line from 1 to 21
        self.current_player = 1  # Players are 1 and -1
        self.done = False
        self.turn_number = 1  # To track the order of claimed numbers

        # Claimed numbers with turn numbers for both players
        self.claimed_numbers_p1 = []  # List of tuples (number, turn_number)
        self.claimed_numbers_p2 = []

        return self.board.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, False, {}  # Game is over

        action_number = action  # Number selected (0-20 corresponds to 1-21)
        if action_number < 0 or action_number > 20:
            return self.board.copy(), -10, True, False, {}  # Invalid action

        # Check if the action is valid
        if self.board[action_number] != 0 or not self.is_action_valid(action_number):
            return self.board.copy(), -10, True, False, {}  # Invalid move

        # Valid action, update the board
        self.board[action_number] = self.current_player

        # Update claimed numbers
        claimed_number = action_number + 1  # Convert index to number
        if self.current_player == 1:
            self.claimed_numbers_p1.append((claimed_number, self.turn_number))
        else:
            self.claimed_numbers_p2.append((claimed_number, self.turn_number))

        self.turn_number += 1

        # Check if the game is over
        if not self.has_valid_moves(1) and not self.has_valid_moves(-1):
            self.done = True
            reward = self.calculate_reward()
            return self.board.copy(), reward, True, False, {}  # Game over

        # Switch current player
        self.current_player *= -1

        return self.board.copy(), 0, False, False, {}  # Continue the game

    def render(self):
        board_str = "Number Line:\n"
        display_line = ""
        for i in range(21):
            num = i + 1
            if self.board[i] == 1:
                display_line += f"[P1]{num} "
            elif self.board[i] == -1:
                display_line += f"[P2]{num} "
            else:
                display_line += f"{num} "
        board_str += display_line.strip() + "\n"

        # Show claimed numbers
        p1_numbers = [num for num, _ in self.claimed_numbers_p1]
        p2_numbers = [num for num, _ in self.claimed_numbers_p2]
        board_str += f"Player 1's Claimed Numbers: {p1_numbers}\n"
        board_str += f"Player 2's Claimed Numbers: {p2_numbers}\n"

        return board_str

    def valid_moves(self):
        valid_moves = []
        for i in range(21):
            if self.board[i] == 0 and self.is_action_valid(i):
                valid_moves.append(i)
        return valid_moves

    def is_action_valid(self, action_number):
        # Check if the action is not blocked for the current player
        blocked_numbers = self.get_blocked_numbers(self.current_player)
        if (action_number + 1) in blocked_numbers:
            return False
        return True

    def get_blocked_numbers(self, player):
        claimed_numbers = (
            self.claimed_numbers_p1 if player == 1 else self.claimed_numbers_p2
        )
        blocked_numbers = set()
        for num, _ in claimed_numbers:
            if num - 1 >= 1:
                blocked_numbers.add(num - 1)
            if num + 1 <= 21:
                blocked_numbers.add(num + 1)
        return blocked_numbers

    def has_valid_moves(self, player):
        for i in range(21):
            if self.board[i] == 0:
                action_number = i
                # Check if the action is not blocked for the player
                blocked_numbers = self.get_blocked_numbers(player)
                if (action_number + 1) not in blocked_numbers:
                    return True
        return False

    def calculate_reward(self):
        # Calculate total scores
        score_p1 = sum(num for num, _ in self.claimed_numbers_p1)
        score_p2 = sum(num for num, _ in self.claimed_numbers_p2)

        if score_p1 > score_p2:
            # Player 1 wins
            winner = 1
        elif score_p2 > score_p1:
            # Player 2 wins
            winner = -1
        else:
            # Tie, apply tiebreaker
            highest_p1 = max(
                self.claimed_numbers_p1, key=lambda x: (x[0], -x[1]), default=(0, 0)
            )
            highest_p2 = max(
                self.claimed_numbers_p2, key=lambda x: (x[0], -x[1]), default=(0, 0)
            )

            if highest_p1[0] > highest_p2[0]:
                winner = 1
            elif highest_p2[0] > highest_p1[0]:
                winner = -1
            else:
                # Same highest number, earlier claimant wins
                if highest_p1[1] < highest_p2[1]:
                    winner = 1
                elif highest_p2[1] < highest_p1[1]:
                    winner = -1
                else:
                    # It's a complete tie
                    winner = 0

        if winner == self.current_player:
            return 1  # Current player wins
        else:
            return 0  # Current player loses or tie
