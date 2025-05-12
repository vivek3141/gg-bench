import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Add 1, 1 - Double
        self.action_space = spaces.Discrete(2)
        # Observation: [current player's score, opponent's score]
        self.observation_space = spaces.Box(low=0, high=10, shape=(2,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.scores = [0, 0]  # Player 0 and Player 1 scores
        self.current_player = 0
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if len(valid_actions) == 0:
            # Current player has no valid moves
            opponent = 1 - self.current_player
            # Check if opponent has valid moves
            self.current_player = opponent
            if len(self.valid_moves()) == 0:
                # Neither player can move, current player loses
                self.done = True
                return self._get_obs(), -1, True, False, {}
            else:
                # Opponent can move, game continues
                return self._get_obs(), 0, False, False, {}

        if action not in valid_actions:
            # Invalid action when valid actions are available
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Perform the action
        if action == 0:  # Add 1
            self.scores[self.current_player] += 1
        elif action == 1:  # Double
            self.scores[self.current_player] *= 2

        # Check if player's score is exactly 10
        if self.scores[self.current_player] == 10:
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch to next player
        self.current_player = 1 - self.current_player
        return self._get_obs(), 0, False, False, {}

    def valid_moves(self):
        current_score = self.scores[self.current_player]
        valid_actions = []
        if current_score + 1 <= 10:
            valid_actions.append(0)  # Add 1
        if current_score * 2 <= 10:
            valid_actions.append(1)  # Double
        return valid_actions

    def _get_obs(self):
        # Observation is [current player's score, opponent's score]
        return np.array(
            [self.scores[self.current_player], self.scores[1 - self.current_player]],
            dtype=np.int32,
        )

    def render(self):
        return (
            "Current Player: Player {}\n" "Player 0 Score: {}\n" "Player 1 Score: {}"
        ).format(self.current_player, self.scores[0], self.scores[1])
