import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 9 claim actions (cells 1-9)
        # There are 225 challenge actions (9 cells * 5 player numbers * 5 opponent numbers)
        # Total actions = 9 + 225 = 234
        self.action_space = spaces.Discrete(234)

        # Observation space:
        # - 9 cells: values -1 (opponent), 0 (empty), 1 (current player)
        # - 5 current player's challenge numbers: 1 (available), 0 (used)
        # - 5 opponent's challenge numbers: 1 (available), 0 (used)
        # Total observation size: 19
        self.observation_space = spaces.Box(low=-1, high=1, shape=(19,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the board
        self.board = np.zeros(9, dtype=np.int8)

        # Challenge numbers: 1 to 5 for each player
        self.player_challenge_numbers = [1, 2, 3, 4, 5]
        self.opponent_challenge_numbers = [1, 2, 3, 4, 5]

        self.current_player = 1  # 1 or -1 (1 for 'X', -1 for 'O')
        self.done = False

        observation = self._get_obs()
        return observation, {}  # Observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), -10, True, False, {}

        reward = 0

        # Handle action
        if action < 9:
            # Claim action
            cell = action  # Cells are 0-indexed
            if self.board[cell] != 0:
                # Invalid move: cell already taken
                self.done = True
                return self._get_obs(), -10, True, False, {}
            else:
                self.board[cell] = self.current_player
        else:
            # Challenge action
            action_id = action - 9
            cell_to_challenge = (action_id // 25) + 0  # 0-indexed
            rest = action_id % 25
            player_challenge_number = (rest // 5) + 1
            opponent_challenge_number = (rest % 5) + 1

            # Check if cell can be challenged
            if (
                self.board[cell_to_challenge] == 0
                or self.board[cell_to_challenge] == self.current_player
            ):
                # Invalid challenge: cell is empty or owned by current player
                self.done = True
                return self._get_obs(), -10, True, False, {}

            # Check if player has the challenge number available
            if player_challenge_number not in self.player_challenge_numbers:
                # Invalid move: challenge number not available
                self.done = True
                return self._get_obs(), -10, True, False, {}

            # Check if opponent has the challenge number available
            if opponent_challenge_number not in self.opponent_challenge_numbers:
                # Invalid move: opponent's challenge number not available
                self.done = True
                return self._get_obs(), -10, True, False, {}

            # Process challenge
            if player_challenge_number > opponent_challenge_number:
                # Current player wins the challenge
                self.board[cell_to_challenge] = self.current_player
            elif player_challenge_number < opponent_challenge_number:
                # Opponent retains the cell (no change needed)
                pass
            else:
                # Tie: Defender retains the cell
                pass

            # Remove used challenge numbers
            self.player_challenge_numbers.remove(player_challenge_number)
            self.opponent_challenge_numbers.remove(opponent_challenge_number)

        # Check for game end condition (all cells claimed)
        if np.all(self.board != 0):
            self.done = True
            # Determine the winner
            player_cells = np.sum(self.board == self.current_player)
            opponent_cells = np.sum(self.board == -self.current_player)
            if player_cells > opponent_cells:
                reward = 1  # Current player wins
            elif player_cells < opponent_cells:
                reward = -1  # Current player loses
            else:
                # Tie-breaker: highest sum of unused challenge numbers
                player_sum = sum(self.player_challenge_numbers)
                opponent_sum = sum(self.opponent_challenge_numbers)
                if player_sum > opponent_sum:
                    reward = 1  # Current player wins
                elif player_sum < opponent_sum:
                    reward = -1  # Current player loses
                else:
                    # Sudden death not implemented; treat as tie
                    reward = 0
        else:
            # Switch players
            self.current_player *= -1
            # Swap challenge numbers for next player
            (
                self.player_challenge_numbers,
                self.opponent_challenge_numbers,
            ) = (
                self.opponent_challenge_numbers,
                self.player_challenge_numbers,
            )

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        marker_map = {0: " ", 1: "X", -1: "O"}
        board_str = ""
        for i in range(3):
            row = "|".join(marker_map[self.board[i * 3 + j]] for j in range(3))
            board_str += row
            if i < 2:
                board_str += "\n-+-+-\n"
        return board_str

    def valid_moves(self):
        valid_actions = []

        # Claim actions
        for i in range(9):
            if self.board[i] == 0:
                valid_actions.append(i)

        # Challenge actions
        opponent_cells = [
            idx for idx, val in enumerate(self.board) if val == -self.current_player
        ]
        for cell in opponent_cells:
            for player_num in self.player_challenge_numbers:
                for opp_num in self.opponent_challenge_numbers:
                    action_id = 9 + cell * 25 + (player_num - 1) * 5 + (opp_num - 1)
                    valid_actions.append(action_id)

        return valid_actions

    def _get_obs(self):
        # Observation includes:
        # - Board state: current player's perspective
        # - Current player's remaining challenge numbers
        # - Opponent's remaining challenge numbers
        board_state = self.current_player * self.board.copy()
        board_state = np.clip(board_state, -1, 1)

        player_nums = np.zeros(5, dtype=np.int8)
        for num in self.player_challenge_numbers:
            player_nums[num - 1] = 1

        opponent_nums = np.zeros(5, dtype=np.int8)
        for num in self.opponent_challenge_numbers:
            opponent_nums[num - 1] = 1

        observation = np.concatenate((board_state, player_nums, opponent_nums))
        return observation
