import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(low=0, high=20, shape=(18,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the grid with numbers 1 to 16, randomly shuffled
        self.grid_numbers = np.arange(1, 17)
        np.random.shuffle(self.grid_numbers)
        self.grid = self.grid_numbers.copy()
        # Initialize player scores
        self.player_scores = [20, 20]
        # Randomly select starting player
        self.current_player = np.random.choice([0, 1])
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        # Validate action
        if action < 0 or action >= 16 or self.grid[action] == 0:
            # Invalid move: selected empty cell or out of bounds
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        selected_number = self.grid[action]
        opponent = 1 - self.current_player

        # Check if selecting this number would reduce opponent's score below zero
        if self.player_scores[opponent] - selected_number < 0:
            # Invalid move: would reduce opponent's score below zero
            self.done = True
            reward = -10
            return self._get_observation(), reward, True, False, {}

        # Apply move
        self.player_scores[opponent] -= selected_number
        self.grid[action] = 0  # Remove number from grid

        # Check for win condition
        if self.player_scores[opponent] == 0:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}

        # Check if opponent has valid moves
        valid = self._opponent_has_valid_moves()
        if not valid:
            # Opponent cannot make a valid move; current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}

        # Switch current player
        self.current_player = opponent
        reward = 0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        grid_str = "\nCurrent Grid:\n"
        for i in range(4):
            grid_str += "|"
            for j in range(4):
                idx = i * 4 + j
                val = self.grid[idx]
                if val == 0:
                    grid_str += " X |"
                else:
                    grid_str += f"{val:>2} |"
            grid_str += "\n"
        grid_str += f"\nPlayer {self.current_player + 1}'s Turn\n"
        grid_str += f"Player 1 Score: {self.player_scores[0]}\n"
        grid_str += f"Player 2 Score: {self.player_scores[1]}\n"
        return grid_str

    def valid_moves(self):
        opponent = 1 - self.current_player
        valid_actions = []
        for action in range(16):
            if self.grid[action] != 0:
                if self.player_scores[opponent] - self.grid[action] >= 0:
                    valid_actions.append(action)
        return valid_actions

    def _get_observation(self):
        # Observation includes the grid and player scores
        observation = np.concatenate((self.grid.copy(), np.array(self.player_scores)))
        return observation

    def _opponent_has_valid_moves(self):
        # Check if opponent has any valid moves
        current_player = self.current_player
        opponent = 1 - current_player
        for number in self.grid:
            if number != 0 and self.player_scores[current_player] - number >= 0:
                return True
        return False
