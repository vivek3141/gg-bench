import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(10,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(
            9, dtype=np.float32
        )  # Numbers 1 to 9 represented by indices 0 to 8
        self.current_player = 1  # Player 1: 1, Player 2: -1
        self.done = False
        observation = np.append(self.board, self.current_player)
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                np.append(self.board, self.current_player),
                0,
                True,
                False,
                {},
            )

        if action < 0 or action >= 9 or self.board[action] != 0:
            self.done = True
            return (
                np.append(self.board, self.current_player),
                -10,
                True,
                False,
                {},
            )

        # Update the board and the player's sequence
        self.board[action] = self.current_player

        # Check for win condition
        player_positions = np.where(self.board == self.current_player)[0]
        player_numbers = player_positions + 1  # Map indices to numbers 1-9

        reward = 0
        self.done = False

        if len(player_numbers) >= 3:
            # Check all combinations of 3 numbers
            for combo in itertools.combinations(player_numbers, 3):
                if self.is_arithmetic_progression(
                    combo
                ) or self.is_geometric_progression(combo):
                    reward = 1
                    self.done = True
                    break

        if not self.done:
            # Check if all numbers have been selected
            if np.all(self.board != 0):
                # Game over, compare sums
                opponent = -self.current_player
                opponent_positions = np.where(self.board == opponent)[0]
                opponent_numbers = opponent_positions + 1

                player_sum = np.sum(player_numbers)
                opponent_sum = np.sum(opponent_numbers)

                if player_sum > opponent_sum:
                    reward = 1
                elif player_sum < opponent_sum:
                    reward = -1
                else:
                    # Sums are equal, last player who took a turn loses
                    reward = -1

                self.done = True
            else:
                # Switch player
                self.current_player *= -1

        observation = np.append(self.board, self.current_player)
        return observation, reward, self.done, False, {}

    def render(self):
        board_str = "Available Numbers:\n"
        for i in range(9):
            if self.board[i] == 0:
                board_str += f"{i + 1} "
            else:
                board_str += "X "
        board_str += "\n\n"
        board_str += f"Player 1's Numbers: {self.get_player_numbers(1)}\n"
        board_str += f"Player 2's Numbers: {self.get_player_numbers(-1)}\n"
        board_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return board_str

    def valid_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def get_player_numbers(self, player):
        positions = np.where(self.board == player)[0]
        numbers = positions + 1  # Map indices to numbers 1-9
        return list(numbers)

    def is_arithmetic_progression(self, nums):
        nums = sorted(nums)
        if nums[1] - nums[0] == nums[2] - nums[1]:
            return True
        return False

    def is_geometric_progression(self, nums):
        nums = sorted(nums)
        if nums[0] == 0 or nums[1] == 0:
            return False
        if nums[1] / nums[0] == nums[2] / nums[1]:
            return True
        return False
