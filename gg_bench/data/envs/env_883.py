import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(
            low=0, high=99, shape=(25, 3), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Generate a 5x5 grid filled with random integers from 1 to 9
        self.grid_numbers = np.random.randint(1, 10, size=(5, 5), dtype=np.int32)
        self.grid_status = np.zeros(
            (5, 5), dtype=np.int32
        )  # 0: available, 1 or 2: selected by player
        self.player_scores = {1: 0, 2: 0}
        self.current_player = 1
        self.last_move_per_player = {1: None, 2: None}
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_moves = self._get_valid_moves(self.current_player)

        if not valid_moves:
            # Current player has no valid moves, skip turn
            opponent = 1 if self.current_player == 2 else 2
            # Check if opponent also has no valid moves
            if not self._get_valid_moves(opponent):
                # Game over
                self.done = True
                if self.player_scores[1] > self.player_scores[2]:
                    reward = 1 if self.current_player == 1 else -1
                elif self.player_scores[2] > self.player_scores[1]:
                    reward = -1 if self.current_player == 1 else 1
                else:
                    reward = 0
                return self._get_observation(), reward, True, False, {}
            else:
                # Switch to opponent's turn
                self.current_player = opponent
                # Return zero reward, game continues
                return self._get_observation(), 0, False, False, {}
        else:
            # Current player has valid moves, process action
            if action not in valid_moves:
                # Invalid move
                return self._get_observation(), -10, True, False, {}

            # Valid move
            row = action // 5
            col = action % 5
            # Update grid status
            self.grid_status[row, col] = self.current_player

            # Update player's score
            cell_value = self.grid_numbers[row, col]
            self.player_scores[self.current_player] += cell_value

            # Update last move
            self.last_move_per_player[self.current_player] = (row, col)

            # Switch to opponent's turn
            opponent = 1 if self.current_player == 2 else 2
            self.current_player = opponent

            # Check if game over
            if not self._get_valid_moves(
                self.current_player
            ) and not self._get_valid_moves(opponent):
                self.done = True
                if self.player_scores[1] > self.player_scores[2]:
                    reward = 1 if self.current_player == 1 else -1
                elif self.player_scores[2] > self.player_scores[1]:
                    reward = -1 if self.current_player == 1 else 1
                else:
                    reward = 0
                return self._get_observation(), reward, True, False, {}
            else:
                return self._get_observation(), 0, False, False, {}

    def render(self):
        grid_str = "\nCurrent Player: Player {}\n".format(self.current_player)
        grid_str += "Player 1 Score: {}\n".format(self.player_scores[1])
        grid_str += "Player 2 Score: {}\n".format(self.player_scores[2])
        grid_str += "-------------------------\n"
        for i in range(5):
            row_str = "|"
            for j in range(5):
                if self.grid_status[i, j] == 0:
                    row_str += " {:2d} |".format(self.grid_numbers[i, j])
                elif self.grid_status[i, j] == 1:
                    row_str += " P1 |"
                elif self.grid_status[i, j] == 2:
                    row_str += " P2 |"
            grid_str += row_str + "\n"
            grid_str += "-------------------------\n"
        return grid_str

    def valid_moves(self):
        return self._get_valid_moves(self.current_player)

    def _get_valid_moves(self, player):
        valid_moves = []
        opponent = 1 if player == 2 else 2
        opponent_last_move = self.last_move_per_player[opponent]
        if opponent_last_move is None:
            # First move, any available cell
            for idx in range(25):
                row = idx // 5
                col = idx % 5
                if self.grid_status[row, col] == 0:
                    valid_moves.append(idx)
        else:
            # Must select an available cell adjacent to opponent's last move
            opp_row, opp_col = opponent_last_move
            for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row = opp_row + delta[0]
                new_col = opp_col + delta[1]
                if 0 <= new_row < 5 and 0 <= new_col < 5:
                    if self.grid_status[new_row, new_col] == 0:
                        idx = new_row * 5 + new_col
                        valid_moves.append(idx)
        return valid_moves

    def _get_observation(self):
        observation = np.zeros((25, 3), dtype=np.int32)
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                observation[idx, 0] = self.grid_numbers[i, j]
                observation[idx, 1] = self.grid_status[i, j]
                if self.last_move_per_player[1] == (i, j) or self.last_move_per_player[
                    2
                ] == (i, j):
                    observation[idx, 2] = 1
                else:
                    observation[idx, 2] = 0
        return observation
