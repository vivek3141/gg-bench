import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Game parameters
        self.target_total = 25

        # Define action and observation space
        # Actions correspond to selecting a number from 1 to 10 (action + 1)
        self.action_space = spaces.Discrete(10)  # Actions: 0 to 9
        # Observation is the current player's total and the opponent's total
        self.observation_space = spaces.Box(
            low=0, high=self.target_total, shape=(2,), dtype=np.int32
        )

        # Initialize state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.totals = [0, 0]  # [Player 1 total, Player 2 total]
        self.done = False
        # Randomly choose starting player (0 or 1)
        self.current_player = self.np_random.integers(2)
        return self.get_observation(), {}  # Observation, info

    def step(self, action):
        if self.done:
            # Game is already over
            return self.get_observation(), -10, True, False, {}

        if not self.action_space.contains(action):
            # Invalid action
            self.done = True
            return self.get_observation(), -10, True, False, {}

        # Map action to selected number (1 to 10)
        selected_number = action + 1  # Actions 0-9 correspond to numbers 1-10

        # Add selected number to opponent's total
        opponent = 1 - self.current_player
        self.totals[opponent] += selected_number

        # Check if opponent's total reaches or exceeds the target total
        if self.totals[opponent] >= self.target_total:
            # Current player wins
            self.done = True
            reward = 1
            return self.get_observation(), reward, True, False, {}
        else:
            # Game continues
            # Switch current player
            self.current_player = opponent
            reward = -10
            return self.get_observation(), reward, False, False, {}

    def render(self):
        return (
            f"Player 1 total: {self.totals[0]}, "
            f"Player 2 total: {self.totals[1]}, "
            f"Current player: Player {self.current_player + 1}"
        )

    def valid_moves(self):
        # All actions from 0 to 9 are valid (numbers 1 to 10)
        return list(range(10))

    def get_observation(self):
        # Observation is [current player's total, opponent's total]
        return np.array(
            [self.totals[self.current_player], self.totals[1 - self.current_player]],
            dtype=np.int32,
        )
