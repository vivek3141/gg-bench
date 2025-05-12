import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_score=15):
        super(CustomEnv, self).__init__()

        self.target_score = target_score

        # Define action and observation space
        # Actions correspond to numbers 1 to 9 (action 0 => number 1, action 8 => number 9)
        self.action_space = spaces.Discrete(9)

        # Observation space contains the scores of the current and opposing player
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.target_score, self.target_score]),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_scores = [0, 0]  # Scores for Player 1 and Player 2
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Observation is the current player's score and the opponent's score
        return np.array(
            [
                self.player_scores[self.current_player],
                self.player_scores[1 - self.current_player],
            ],
            dtype=np.int32,
        )

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        if action not in self.valid_moves():
            self.done = True
            return self._get_obs(), -10, True, False, {}

        number_chosen = action + 1  # Map action 0-8 to numbers 1-9
        self.player_scores[self.current_player] += number_chosen

        # Check for win condition
        if self.player_scores[self.current_player] == self.target_score:
            self.done = True
            return self._get_obs(), 1, True, False, {}  # Current player wins

        # Check for loss condition
        if self.player_scores[self.current_player] > self.target_score:
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Current player loses

        # Valid move, switch turns
        reward = 0  # No reward for a valid move
        self.current_player = 1 - self.current_player
        return self._get_obs(), reward, False, False, {}

    def render(self):
        return (
            f"Target Score: {self.target_score}\n"
            f"Player {self.current_player + 1}'s Turn\n"
            f"Player 1 Score: {self.player_scores[0]}\n"
            f"Player 2 Score: {self.player_scores[1]}\n"
        )

    def valid_moves(self):
        # All actions from 0 to 8 are valid (numbers 1 to 9)
        return list(range(9))
