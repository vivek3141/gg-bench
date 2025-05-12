import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: 0 - Add 1, 1 - Multiply by 2
        self.action_space = spaces.Discrete(2)

        # Observation space: [target_number, current_player, player0_score, player1_score]
        self.observation_space = spaces.Box(
            low=np.array([15, 0, 0, 0]), high=np.array([25, 1, 25, 25]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_number = np.random.randint(
            15, 26
        )  # Target Number between 15 and 25 inclusive
        self.player_scores = [0, 0]
        self.current_player = 0  # Player indices: 0 or 1
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            # The game is over
            return self._get_obs(), 0, True, False, {}

        if not self.action_space.contains(action):
            # Invalid action
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Apply the action
        current_score = self.player_scores[self.current_player]
        if action == 0:
            # Add 1
            new_score = current_score + 1
        else:  # action == 1
            # Multiply by 2
            new_score = current_score * 2

        # Update the player's score
        self.player_scores[self.current_player] = new_score

        if new_score > self.target_number:
            # Current player loses
            self.done = True
            reward = -1
            return self._get_obs(), reward, True, False, {}
        elif new_score == self.target_number:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}
        else:
            # Game continues
            self.current_player = 1 - self.current_player  # Switch player
            reward = 0
            return self._get_obs(), reward, False, False, {}

    def render(self):
        return (
            f"Target Number: {self.target_number}\n"
            f"Player 0 Score: {self.player_scores[0]}\n"
            f"Player 1 Score: {self.player_scores[1]}\n"
            f"Current Player: {self.current_player}\n"
        )

    def valid_moves(self):
        current_score = self.player_scores[self.current_player]
        valid_actions = []
        # Check if adding 1 is a valid move
        if current_score + 1 <= self.target_number:
            valid_actions.append(0)
        # Check if multiplying by 2 is a valid move
        if current_score * 2 <= self.target_number:
            valid_actions.append(1)
        return valid_actions

    def _get_obs(self):
        return np.array(
            [
                self.target_number,
                self.current_player,
                self.player_scores[0],
                self.player_scores[1],
            ],
            dtype=np.int32,
        )
