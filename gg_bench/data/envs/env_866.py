import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: positions 0 to 8 (9 positions)
        self.action_space = spaces.Discrete(9)

        # Define observation space: grid of 9 positions, values from 0 to 9 (0 indicates empty)
        self.observation_space = spaces.Box(low=0, high=9, shape=(9,), dtype=np.int32)

        # Initialize other variables
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize grid with numbers 1 to 9, randomly assigned
        self.grid_numbers = np.arange(1, 10)
        self.np_random.shuffle(self.grid_numbers)

        # Grid status: 1 means available, 0 means captured
        self.grid_status = np.ones(9, dtype=np.int32)

        # Current player: 1 for player 1, -1 for player 2
        self.current_player = 1

        # Players' scores
        self.player_scores = {1: 0, -1: 0}

        # Done flag
        self.done = False

        # Prepare the observation
        observation = self._get_observation()

        return observation, {}  # Return observation and info

    def step(self, action):
        # Validate the action
        if self.done:
            # Game is over
            observation = self._get_observation()
            return observation, -10, True, {}  # observation, reward, done, info
        if action < 0 or action >= 9 or self.grid_status[action] == 0:
            # Invalid move
            observation = self._get_observation()
            self.done = True
            return observation, -10, True, {}  # observation, reward, done, info

        # Valid move
        # Capture the selected square
        captured_squares = [action]
        self.grid_status[action] = 0
        selected_number = self.grid_numbers[action]

        # Capture eligible adjacent squares
        adjacent_indices = self._get_adjacent_indices(action)
        for idx in adjacent_indices:
            if self.grid_status[idx] == 1 and self.grid_numbers[idx] < selected_number:
                captured_squares.append(idx)
                self.grid_status[idx] = 0

        # Update player's score
        turn_score = sum(self.grid_numbers[idx] for idx in captured_squares)
        self.player_scores[self.current_player] += turn_score

        # Check if the game is over
        if np.all(self.grid_status == 0):
            self.done = True
            # Determine winner
            player_score = self.player_scores[self.current_player]
            opponent_score = self.player_scores[-self.current_player]
            if player_score > opponent_score:
                # Current player wins
                reward = 1
            else:
                # Current player loses
                reward = -1
            observation = self._get_observation()
            return observation, reward, True, {}  # observation, reward, done, info
        else:
            # Game continues
            # Switch player
            self.current_player *= -1
            observation = self._get_observation()
            return observation, -10, False, {}  # observation, reward, done, info

    def _get_observation(self):
        # The observation is the grid_numbers where the status is 1, 0 where status is 0
        observation = self.grid_numbers * self.grid_status
        return observation

    def _get_adjacent_indices(self, index):
        # Given the index (0-8), return the indices of orthogonally adjacent squares
        adjacent_indices = []
        row = index // 3
        col = index % 3
        # Up
        if row > 0:
            adjacent_indices.append((row - 1) * 3 + col)
        # Down
        if row < 2:
            adjacent_indices.append((row + 1) * 3 + col)
        # Left
        if col > 0:
            adjacent_indices.append(row * 3 + col - 1)
        # Right
        if col < 2:
            adjacent_indices.append(row * 3 + col + 1)
        return adjacent_indices

    def render(self):
        # Visual representation of the grid
        grid_str = ""
        for i in range(3):
            grid_str += "+---+---+---+\n"
            for j in range(3):
                idx = i * 3 + j
                if self.grid_status[idx] == 1:
                    value = self.grid_numbers[idx]
                else:
                    value = " "
                grid_str += f"| {value} "
            grid_str += "|\n"
        grid_str += "+---+---+---+\n"
        return grid_str

    def valid_moves(self):
        return [idx for idx in range(9) if self.grid_status[idx] == 1]
