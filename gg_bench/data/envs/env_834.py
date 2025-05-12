import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(9)
        # Observation space:
        # Index 0-8: Availability of numbers 1-9 in the shared pool (1 if available, 0 if not)
        # Index 9: Current player's cumulative score
        # Index 10: Opponent's cumulative score
        self.observation_space = spaces.Box(low=0, high=15, shape=(11,), dtype=np.int8)

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Shared pool: 1 if number (index + 1) is available, 0 if not
        self.shared_pool = np.ones(9, dtype=np.int8)
        # Player scores: [player 0 score, player 1 score]
        self.player_scores = [0, 0]
        # Current player: 0 or 1
        self.current_player = 0
        self.done = False

        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if action is valid
        if action < 0 or action >= 9 or self.shared_pool[action] == 0:
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Invalid move

        # Valid move: update game state
        selected_number = action + 1
        self.player_scores[self.current_player] += selected_number
        self.shared_pool[action] = 0  # Remove the number from the pool

        reward = 0
        # Check for winning condition
        if self.player_scores[self.current_player] == 15:
            reward = 1
            self.done = True
        elif self.player_scores[self.current_player] > 15:
            self.done = True
        elif np.sum(self.shared_pool) == 0:
            # No numbers left in the pool
            self.done = True
            opponent = 1 - self.current_player
            current_score = self.player_scores[self.current_player]
            opponent_score = self.player_scores[opponent]

            # Determine the winner
            if current_score <= 15 and opponent_score <= 15:
                if current_score > opponent_score:
                    reward = 1
                else:
                    reward = 0
            elif current_score <= 15 and opponent_score > 15:
                reward = 1
            else:
                reward = 0
        else:
            # Switch to the next player
            self.current_player = 1 - self.current_player

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        pool_numbers = [str(i + 1) for i in range(9) if self.shared_pool[i] == 1]
        pool_str = "Shared Pool: [" + ", ".join(pool_numbers) + "]"
        scores_str = (
            f"Player {self.current_player + 1}'s turn.\n"
            f"Player 1 Score: {self.player_scores[0]}\n"
            f"Player 2 Score: {self.player_scores[1]}\n"
        )
        return pool_str + "\n" + scores_str

    def valid_moves(self):
        return [i for i in range(9) if self.shared_pool[i] == 1]

    def _get_obs(self):
        obs = np.zeros(11, dtype=np.int8)
        obs[:9] = self.shared_pool
        obs[9] = self.player_scores[self.current_player]
        obs[10] = self.player_scores[1 - self.current_player]
        return obs
