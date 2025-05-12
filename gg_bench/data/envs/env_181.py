import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space
        self.action_space = spaces.Discrete(9)

        # Define observation space
        # Observation is a vector of length 20:
        # - First 9: Numbers on the grid (1 to 9)
        # - Next 9: Claim status of each cell (0: unclaimed, 1: Player 1, 2: Player 2)
        # - Next 1: Current player's score
        # - Next 1: Opponent's score

        self.observation_space = spaces.Box(low=0, high=45, shape=(20,), dtype=np.int8)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomly assign numbers 1-9 to grid positions
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.grid_numbers = np.arange(1, 10)
        self.np_random.shuffle(self.grid_numbers)

        # Claim status for each cell (0: unclaimed, 1: Player 1, 2: Player 2)
        self.grid_claims = np.zeros(9, dtype=np.int8)

        # Initialize scores
        self.scores = {1: 0, 2: 0}

        # Player 1 starts
        self.current_player = 1

        # Game not over
        self.done = False

        # Return initial observation and info
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            # If game is already over, no further actions are valid
            return self._get_observation(), 0, True, False, {}

        # Invalid move if action is out of bounds or cell is already claimed
        if action < 0 or action >= 9 or self.grid_claims[action] != 0:
            self.done = True
            return self._get_observation(), -10, True, False, {"invalid_move": True}

        # Claim the number
        self.grid_claims[action] = self.current_player

        # Update current player's score
        self.scores[self.current_player] += self.grid_numbers[action]

        # Check if all numbers have been claimed
        if np.all(self.grid_claims != 0):
            self.done = True

            # Determine winner
            if self.scores[self.current_player] > self.scores[3 - self.current_player]:
                # Current player wins
                reward = 1
            else:
                # Current player loses
                reward = 0
            return self._get_observation(), reward, True, False, {}
        else:
            # Switch to the other player
            self.current_player = 3 - self.current_player
            return self._get_observation(), 0, False, False, {}

    def render(self):
        # Generate a visual representation of the grid
        grid_output = ""
        for i in range(3):
            row = ""
            for j in range(3):
                idx = i * 3 + j
                num = self.grid_numbers[idx]
                claim = self.grid_claims[idx]
                if claim == 0:
                    cell_str = f"[{num}]"
                elif claim == 1:
                    cell_str = "[P1]"
                elif claim == 2:
                    cell_str = "[P2]"
                row += f"{cell_str} "
            grid_output += row.strip() + "\n"
        grid_output += f"Current Player: P{self.current_player}\n"
        grid_output += f"Scores - P1: {self.scores[1]}, P2: {self.scores[2]}"
        return grid_output

    def valid_moves(self):
        # Return list of indices of unclaimed cells
        return [i for i in range(9) if self.grid_claims[i] == 0]

    def _get_observation(self):
        # Combine grid numbers, claims, and scores into a single observation
        observation = np.concatenate(
            [
                self.grid_numbers,
                self.grid_claims,
                [
                    self.scores[self.current_player],
                    self.scores[3 - self.current_player],
                ],
            ]
        ).astype(np.int8)
        return observation
