import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 10 digits * 2 players * 4 positions = 80 possible actions
        self.action_space = spaces.Discrete(80)

        # Observation space:
        # - Digit pool counts: indices 0-9 (counts of digits 0-9 in the pool)
        # - Current player's board: indices 10-13 (digits or -1 if empty)
        # - Opponent's board: indices 14-17 (digits or -1 if empty)
        self.observation_space = spaces.Box(
            low=np.array([0] * 10 + [-1] * 8),
            high=np.array([2] * 10 + [9] * 8),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize digit pool: counts of digits 0-9 (each appears twice)
        self.digit_pool = np.array([2] * 10, dtype=np.int32)

        # Initialize players' boards: -1 indicates empty slot
        self.boards = [
            np.array([-1] * 4, dtype=np.int32),  # Player 1's board
            np.array([-1] * 4, dtype=np.int32),
        ]  # Player 2's board

        self.first_player = 0  # Player 0 starts first
        self.current_player = self.first_player
        self.done = False

        # No past action yet
        self.previous_player = None

        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Map action index to (digit, target player, position)
        digit = action // 8
        remainder = action % 8
        target_player = remainder // 4  # 0 for own board, 1 for opponent's board
        position = remainder % 4  # 0: Thousands, 1: Hundreds, 2: Tens, 3: Ones

        # Check if digit is available in the pool
        if digit < 0 or digit > 9 or self.digit_pool[digit] <= 0:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Determine target player index
        if target_player == 0:
            target_player_index = self.current_player
        else:
            target_player_index = 1 - self.current_player

        # Check if target position is empty
        if self.boards[target_player_index][position] != -1:
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Valid move: Assign digit to the target position
        self.boards[target_player_index][position] = digit
        self.digit_pool[digit] -= 1

        # Check if the game is over (all positions filled)
        if np.all(self.boards[0] != -1) and np.all(self.boards[1] != -1):
            self.done = True
            # Calculate final numbers
            player_numbers = []
            for board in self.boards:
                number = (
                    (board[0] * 1000) + (board[1] * 100) + (board[2] * 10) + board[3]
                )
                player_numbers.append(number)

            # Determine winner
            if (
                player_numbers[self.current_player]
                < player_numbers[1 - self.current_player]
            ):
                reward = 1  # Current player wins
            elif (
                player_numbers[self.current_player]
                > player_numbers[1 - self.current_player]
            ):
                reward = -1  # Opponent wins
            else:
                # Tie-breaker: Second player wins
                second_player = 1 - self.first_player
                if self.current_player == second_player:
                    reward = 1  # Current player wins
                else:
                    reward = -1  # Opponent wins
            return self._get_obs(), reward, True, False, {}

        # Valid move, game continues
        reward = 0

        # Prepare for next turn
        self.previous_player = self.current_player
        self.current_player = 1 - self.current_player

        return self._get_obs(), reward, False, False, {}

    def render(self):
        board_str = "\nCurrent Game State:\n"
        board_str += "\nDigit Pool:\n"
        board_str += "Digit:   " + " ".join(str(i) for i in range(10)) + "\n"
        board_str += (
            "Count:   " + " ".join(str(self.digit_pool[i]) for i in range(10)) + "\n"
        )

        for idx, board in enumerate(self.boards):
            player = "Player 1" if idx == 0 else "Player 2"
            board_str += f"\n{player}'s Board:\n"
            positions = ["Thousands", "Hundreds", "Tens", "Ones"]
            board_values = [str(board[i]) if board[i] != -1 else "_" for i in range(4)]
            board_str += "+----------+\n"
            board_str += "| " + " ".join(board_values) + " |\n"
            board_str += "+----------+\n"
            board_str += "  T H T O  \n"

        return board_str

    def valid_moves(self):
        valid_actions = []
        for digit in range(10):
            if self.digit_pool[digit] > 0:
                for target_player in [0, 1]:
                    if target_player == 0:
                        target_player_index = self.current_player
                    else:
                        target_player_index = 1 - self.current_player
                    for position in range(4):
                        if self.boards[target_player_index][position] == -1:
                            action = digit * 8 + target_player * 4 + position
                            valid_actions.append(action)
        return valid_actions

    def _get_obs(self):
        # Construct observation
        current_board = self.boards[self.current_player]
        opponent_board = self.boards[1 - self.current_player]
        observation = np.concatenate((self.digit_pool, current_board, opponent_board))
        return observation
