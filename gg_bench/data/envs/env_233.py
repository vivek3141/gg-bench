import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 8 possible actions, representing multipliers 2 to 9
        self.action_space = spaces.Discrete(
            8
        )  # actions 0 to 7 correspond to multipliers 2 to 9 inclusive

        # Define observation space: [shared_number, current_player]
        # shared_number between 1 and a high limit (e.g., 1e5)
        # current_player: -1 or 1
        self.observation_space = spaces.Box(
            low=np.array([1, -1]), high=np.array([1e5, 1]), dtype=np.float64
        )

        # Set the target number
        self.target_number = 100

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.info = {}
        observation = self._get_obs()
        return observation, self.info  # Return initial observation and info

    def step(self, action):
        if self.done:
            # The game has ended, but step is called again
            return self._get_obs(), 0, True, False, self.info

        if not self.action_space.contains(action):
            # Invalid action, current player loses
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, self.info

        multiplier = action + 2  # Map action to multiplier between 2 and 9 inclusive

        # Update shared number
        self.shared_number *= multiplier

        if self.shared_number >= self.target_number:
            # Current player loses
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, self.info
        else:
            # Game continues, switch player
            self.current_player *= -1  # Switch player
            reward = 0
            return self._get_obs(), reward, False, False, self.info

    def _get_obs(self):
        return np.array([self.shared_number, self.current_player], dtype=np.float64)

    def render(self):
        return f"Shared number: {self.shared_number}, Current player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        # All actions from 0 to 7 are valid (representing multipliers 2 to 9)
        return list(range(8))
