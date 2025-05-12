import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(low=-9, high=19, shape=(25,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid with random integers from 1 to 9 (inclusive)
        self.board = np.random.randint(1, 10, size=(5, 5), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.scores = {1: 0, 2: 0}
        self.done = False
        self.first_move_made = {1: False, 2: False}
        return self.get_observation(), {}  # Return observation and info

    def get_observation(self):
        # Flatten the board to shape (25,)
        obs = self.board.flatten()
        return obs

    def step(self, action):
        if self.done:
            return self.get_observation(), -10, True, False, {}
        row = action // 5
        col = action % 5

        if not (0 <= row < 5 and 0 <= col < 5):
            return self.get_observation(), -10, True, False, {}

        cell_value = self.board[row, col]
        if cell_value > 10 or cell_value < -9:
            # Cell already claimed
            return self.get_observation(), -10, True, False, {}

        valid_moves = self.get_valid_moves()
        if action not in valid_moves:
            # Invalid move as per the rules
            return self.get_observation(), -10, True, False, {}

        # Claim the cell
        self.scores[self.current_player] += cell_value

        # Update the board
        if self.current_player == 1:
            self.board[row, col] = 10 + cell_value  # Claimed by Player 1 (values 11-19)
        else:
            self.board[row, col] = -cell_value  # Claimed by Player 2 (values -1 to -9)

        self.first_move_made[self.current_player] = True

        # Check if game is over
        no_valid_moves_p1 = len(self.get_valid_moves_for_player(1)) == 0
        no_valid_moves_p2 = len(self.get_valid_moves_for_player(2)) == 0
        if no_valid_moves_p1 and no_valid_moves_p2:
            self.done = True
            # Determine winner
            score_p1 = self.scores[1]
            score_p2 = self.scores[2]
            if score_p1 > score_p2:
                winner = 1
            elif score_p2 > score_p1:
                winner = 2
            else:
                # Last player to make a move wins
                winner = self.current_player
            if winner == self.current_player:
                reward = 1
            else:
                reward = 0
            return self.get_observation(), reward, True, False, {}
        else:
            # Switch player
            self.current_player = 2 if self.current_player == 1 else 1
            reward = 0
            return self.get_observation(), reward, False, False, {}

    def get_valid_moves(self):
        return self.get_valid_moves_for_player(self.current_player)

    def get_valid_moves_for_player(self, player):
        valid_moves = []
        # If first move for player
        if not self.first_move_made[player]:
            # Can claim any unclaimed cell on the outer edge
            for row in [0, 4]:
                for col in range(5):
                    if 1 <= self.board[row, col] <= 9:
                        action = row * 5 + col
                        valid_moves.append(action)
            for row in range(1, 4):
                for col in [0, 4]:
                    if 1 <= self.board[row, col] <= 9:
                        action = row * 5 + col
                        valid_moves.append(action)
        else:
            # Can claim any unclaimed cell adjacent to cells already claimed by player
            # Find all positions claimed by player
            if player == 1:
                claimed_positions = np.argwhere((self.board >= 11) & (self.board <= 19))
            else:
                claimed_positions = np.argwhere((self.board <= -1) & (self.board >= -9))
            for pos in claimed_positions:
                row, col = pos
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r, c = row + dr, col + dc
                    if 0 <= r < 5 and 0 <= c < 5:
                        if 1 <= self.board[r, c] <= 9:
                            action = r * 5 + c
                            if action not in valid_moves:
                                valid_moves.append(action)
        return valid_moves

    def render(self):
        board_str = "     1    2    3    4    5\n   +----+----+----+----+----+\n"
        for i in range(5):
            row_str = f"{i+1}  |"
            for j in range(5):
                cell_value = self.board[i, j]
                if 1 <= cell_value <= 9:
                    # Unclaimed cell
                    row_str += f" {cell_value}  |"
                elif 11 <= cell_value <= 19:
                    # Claimed by Player 1
                    row_str += " P1 |"
                elif -9 <= cell_value <= -1:
                    # Claimed by Player 2
                    row_str += " P2 |"
                else:
                    row_str += " ?? |"  # Should not happen
            board_str += row_str + "\n   +----+----+----+----+----+\n"
        scores_str = (
            f"Player 1 score: {self.scores[1]}, Player 2 score: {self.scores[2]}\n"
        )
        current_player_str = f"Current player: Player {self.current_player}\n"
        return board_str + scores_str + current_player_str

    def valid_moves(self):
        return self.get_valid_moves()
