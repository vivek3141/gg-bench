import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 0-15 for squares, 16 for 'pass'
        self.action_space = spaces.Discrete(17)

        # Observation space: 4x4x2 array
        # First channel: values (1-5)
        # Second channel: ownership (-1, 0, 1)
        low_values = np.ones((4, 4), dtype=np.int32)
        high_values = np.full((4, 4), 5, dtype=np.int32)
        low_ownership = np.full((4, 4), -1, dtype=np.int32)
        high_ownership = np.full((4, 4), 1, dtype=np.int32)
        low = np.stack((low_values, low_ownership), axis=-1)
        high = np.stack((high_values, high_ownership), axis=-1)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Initialize the board
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.values = np.random.randint(1, 6, size=(4, 4), dtype=np.int32)
        self.ownership = np.zeros(
            (4, 4), dtype=np.int32
        )  # 0: unclaimed, 1: Player 1, -1: Player 2

        self.player_scores = {1: 0, -1: 0}
        self.claimed_squares = {1: [], -1: []}
        self.current_player = 1  # 1: Player 1, -1: Player 2
        self.done = False
        self.passed_last_turn = {1: False, -1: False}

        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, self.done, {}

        # Check if action is pass
        if action == 16:
            # Player chooses to pass
            valid_moves = self.valid_moves()
            if len(valid_moves) > 0 and 16 not in valid_moves:
                # Passing is invalid if player has valid moves
                self.done = True
                return self._get_observation(), -10, self.done, {}

            # Valid pass
            self.passed_last_turn[self.current_player] = True
            if self.passed_last_turn[1] and self.passed_last_turn[-1]:
                # Both players have passed consecutively
                self.done = True
                score1 = self.player_scores[1] if self.player_scores[1] <= 15 else 0
                score2 = self.player_scores[-1] if self.player_scores[-1] <= 15 else 0

                if score1 > score2:
                    winner = 1
                elif score2 > score1:
                    winner = -1
                else:
                    # Scores equal, Player 2 wins
                    winner = -1

                reward = 1 if winner == self.current_player else 0
                return self._get_observation(), reward, self.done, {}

            else:
                # Switch player
                self.current_player *= -1
                return self._get_observation(), 0, self.done, {}

        else:
            row, col = divmod(action, 4)

            if self.ownership[row, col] != 0:
                # Invalid move
                self.done = True
                return self._get_observation(), -10, self.done, {}

            total_score = (
                self.player_scores[self.current_player] + self.values[row, col]
            )
            if total_score > 15:
                # Cannot exceed 15
                self.done = True
                return self._get_observation(), -10, self.done, {}

            claimed_squares = self.claimed_squares[self.current_player]
            if not claimed_squares:
                # First move
                valid = True
            else:
                # Check adjacency
                valid = False
                for cs_row, cs_col in claimed_squares:
                    if (abs(cs_row - row) == 1 and cs_col == col) or (
                        abs(cs_col - col) == 1 and cs_row == row
                    ):
                        valid = True
                        break
                if not valid:
                    # Invalid move due to adjacency
                    self.done = True
                    return self._get_observation(), -10, self.done, {}

            # Valid move
            self.ownership[row, col] = self.current_player
            self.player_scores[self.current_player] += self.values[row, col]
            self.claimed_squares[self.current_player].append((row, col))
            self.passed_last_turn[self.current_player] = False

            if self.player_scores[self.current_player] == 15:
                # Current player wins
                self.done = True
                return self._get_observation(), 1, self.done, {}
            else:
                self.current_player *= -1
                return self._get_observation(), 0, self.done, {}

    def render(self):
        board_str = ""
        for i in range(4):
            row_str = ""
            for j in range(4):
                value = self.values[i, j]
                owner = self.ownership[i, j]
                if owner == 1:
                    cell = f" P1({value}) "
                elif owner == -1:
                    cell = f" P2({value}) "
                else:
                    cell = f"  {value}   "
                row_str += cell
            board_str += row_str + "\n"
        return board_str

    def _get_observation(self):
        observation = np.stack((self.values, self.ownership), axis=-1)
        return observation

    def valid_moves(self):
        valid_actions = []

        for action in range(16):
            row, col = divmod(action, 4)
            if self.ownership[row, col] != 0:
                continue  # Square already claimed

            total_score = (
                self.player_scores[self.current_player] + self.values[row, col]
            )
            if total_score > 15:
                continue  # Exceeds maximum score

            claimed_squares = self.claimed_squares[self.current_player]
            if not claimed_squares:
                # First move, any unclaimed square is valid
                valid_actions.append(action)
            else:
                # Need to check adjacency
                for cs_row, cs_col in claimed_squares:
                    if (abs(cs_row - row) == 1 and cs_col == col) or (
                        abs(cs_col - col) == 1 and cs_row == row
                    ):
                        valid_actions.append(action)
                        break

        if not valid_actions:
            valid_actions.append(16)  # Action 16 is 'pass'

        return valid_actions
