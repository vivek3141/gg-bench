import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # There are 25 cells in a 5x5 grid
        self.action_space = spaces.Discrete(25)
        # Define the observation space
        self.observation_space = spaces.Dict(
            {
                "board_status": spaces.Box(low=-1, high=3, shape=(5, 5), dtype=np.int8),
                "cell_values": spaces.Box(low=0, high=5, shape=(5, 5), dtype=np.int8),
                "scores": spaces.Box(low=0, high=100, shape=(2,), dtype=np.int8),
            }
        )
        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly assign cell values between 1 and 5
        self.cell_values = np.random.randint(1, 6, size=(5, 5), dtype=np.int8)
        # Initialize board status to -1 (hidden)
        self.board_status = np.full((5, 5), -1, dtype=np.int8)
        # Initialize scores
        self.scores = np.array([0, 0], dtype=np.int8)
        # Set current player (0 or 1)
        self.current_player = 0
        # Game not done
        self.done = False
        # Not in sudden death
        self.sudden_death = False
        return self._get_observation(), {}

    def step(self, action):
        if self.done:
            # Return the current observation with no change
            return self._get_observation(), 0, True, False, {}
        # Map action to cell indices
        i = action // 5
        j = action % 5
        # Check if action is valid
        if self.board_status[i, j] in [1, 2]:
            # Invalid move, cell already claimed
            self.done = True
            return self._get_observation(), -10, True, False, {"Invalid": True}
        elif self.board_status[i, j] in [-1, 0]:
            # Valid move
            # Claim the cell
            self.board_status[i, j] = (
                self.current_player + 1
            )  # 1 for Player 1, 2 for Player 2
            # Add cell's value to current player's score
            self.scores[self.current_player] += self.cell_values[i, j]
            # Reveal adjacent unclaimed cells
            self._reveal_adjacent_cells(i, j)
            # Check for game end conditions
            reward = 0
            info = {"Invalid": False}
            if self.scores[self.current_player] >= 15:
                # Current player wins
                self.done = True
                reward = 1
            elif np.all(self.board_status >= 1):
                # All cells are claimed
                if self.scores[0] == self.scores[1]:
                    # Tie, enter sudden death
                    self.sudden_death = True
                else:
                    # Player with higher score wins
                    self.done = True
                    winner = np.argmax(self.scores)
                    if self.current_player == winner:
                        reward = 1
                    else:
                        reward = 0
            elif self.sudden_death:
                if self.scores[0] != self.scores[1]:
                    # Player with higher score wins
                    self.done = True
                    winner = np.argmax(self.scores)
                    if self.current_player == winner:
                        reward = 1
                    else:
                        reward = 0
            # Switch current player
            if not self.done:
                self.current_player = 1 - self.current_player
            return self._get_observation(), reward, self.done, False, info
        else:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {"Invalid": True}

    def _reveal_adjacent_cells(self, i, j):
        # Reveal all orthogonally and diagonally adjacent unclaimed cells
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 5 and 0 <= nj < 5:
                    if self.board_status[ni, nj] == -1:
                        self.board_status[ni, nj] = 0  # Revealed

    def _get_observation(self):
        # Return the observation as a dictionary
        return {
            "board_status": self.board_status.copy(),
            "cell_values": np.where(self.board_status >= 0, self.cell_values, 0),
            "scores": self.scores.copy(),
        }

    def render(self):
        # Build a string representation of the grid
        grid_str = ""
        for i in range(5):
            for j in range(5):
                status = self.board_status[i, j]
                if status == -1:
                    cell_str = "[ ]"
                elif status == 0:
                    cell_str = f"[{self.cell_values[i, j]} ]"
                elif status == 1:
                    cell_str = "[P1]"
                elif status == 2:
                    cell_str = "[P2]"
                grid_str += cell_str + " "
            grid_str += "\n"
        # Append the scores
        scores_str = (
            f"Player 1 score: {self.scores[0]}\nPlayer 2 score: {self.scores[1]}\n"
        )
        return grid_str + scores_str

    def valid_moves(self):
        # Return a list of valid actions (unclaimed cells)
        valid_actions = []
        for i in range(5):
            for j in range(5):
                if self.board_status[i, j] in [-1, 0]:
                    action = i * 5 + j
                    valid_actions.append(action)
        return valid_actions
