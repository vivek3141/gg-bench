import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: numbers from 1 to 10 (actions 0 to 9)
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # Index 0: Cumulative total (0 to 50)
        # Indices 1-10: Availability of numbers 1 to 10 (1 if available, 0 if used)
        self.observation_space = spaces.Box(
            low=np.array([0] + [0] * 10), high=np.array([50] + [1] * 10), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_total = 0
        self.numbers_available = np.ones(10, dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}

    def step(self, action):
        reward = -10  # Default reward for a valid move

        if self.done:
            return self._get_obs(), reward, True, False, {}
        elif action < 0 or action >= 10 or self.numbers_available[action] == 0:
            # Invalid action
            reward = -10
            self.done = True
            return self._get_obs(), reward, True, False, {}
        else:
            number_chosen = action + 1  # Numbers 1 to 10
            self.cumulative_total += number_chosen
            self.numbers_available[action] = 0  # Mark the number as used

            # Check if cumulative total reaches or exceeds 50
            if self.cumulative_total >= 50:
                # Current player loses
                self.done = True
                return self._get_obs(), reward, True, False, {}

            # Check if opponent has any valid moves without exceeding 50
            opponent_can_move = False
            for i in range(10):
                if self.numbers_available[i] == 1:
                    if self.cumulative_total + (i + 1) < 50:
                        opponent_can_move = True
                        break
            if not opponent_can_move:
                # Current player wins
                reward = 1
                self.done = True
                return self._get_obs(), reward, True, False, {}
            else:
                # Game continues
                self.current_player = 3 - self.current_player  # Switch player
                return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        # Observation: [cumulative_total, numbers_available...]
        return np.concatenate(([self.cumulative_total], self.numbers_available))

    def render(self):
        numbers_available = [i + 1 for i in range(10) if self.numbers_available[i] == 1]
        return (
            f"Cumulative Total: {self.cumulative_total}\n"
            f"Available Numbers: {numbers_available}\n"
            f"Current Player: Player {self.current_player}"
        )

    def valid_moves(self):
        return [i for i in range(10) if self.numbers_available[i] == 1]
