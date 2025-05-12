import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Discrete(9) for numbers 1 to 9 mapped to 0 to 8
        self.action_space = spaces.Discrete(9)

        # Observation space: Box(0, 50, shape=(2,)) to represent the scores of both players
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([50, 50]), dtype=np.int32
        )

        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.scores = [0, 0]  # Scores for Player 1 and Player 2
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 0  # Player 1 starts
        self.scores = [0, 0]
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def step(self, action):
        if self.done or not self.action_space.contains(action):
            # Invalid move
            return self._get_obs(), -10, True, False, {}

        # Map action (0-8) to selected number (1-9)
        selected_number = action + 1

        opponent = 1 - self.current_player

        # Check if the selected number is odd or even
        if selected_number % 2 == 1:
            # Odd number: Add to current player's score
            self.scores[self.current_player] += selected_number
        else:
            # Even number: Add to opponent's score
            self.scores[opponent] += selected_number

        reward = 0

        # Check for win/loss conditions
        # Current player wins
        if self.scores[self.current_player] == 25:
            self.done = True
            reward = 1
        # Current player loses by exceeding 25
        elif self.scores[self.current_player] > 25:
            self.done = True
            reward = -1
        # Opponent wins
        elif self.scores[opponent] == 25:
            self.done = True
            reward = -1
        # Opponent loses by exceeding 25
        elif self.scores[opponent] > 25:
            self.done = True
            reward = 1
        else:
            # Continue the game and switch players
            self.current_player = opponent

        return (
            self._get_obs(),
            reward,
            self.done,
            False,
            {},
        )  # Observation, reward, done, truncated, info

    def render(self):
        # Visual representation of the current state
        return (
            f"Player 1 Score: {self.scores[0]}\n"
            f"Player 2 Score: {self.scores[1]}\n"
            f"Current Player: {'Player 1' if self.current_player == 0 else 'Player 2'}"
        )

    def valid_moves(self):
        # All moves are valid unless the game is done
        return list(range(9)) if not self.done else []

    def _get_obs(self):
        # Observation is an array of the two players' scores
        return np.array(self.scores, dtype=np.int32)
