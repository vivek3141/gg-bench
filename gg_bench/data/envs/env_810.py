import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: numbers from 1 to 9 represented as actions 0 to 8
        self.action_space = spaces.Discrete(9)

        # Observation space:
        # - obs[0]: Total (0 to 31)
        # - obs[1-9]: Available Numbers indicator (1 if available, 0 if not)
        self.observation_space = spaces.Box(low=0, high=31, shape=(10,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total = 0
        self.available_numbers = np.ones(9, dtype=np.int32)  # Numbers 1-9 are available
        self.current_player = 1  # Players are 1 and -1; agent plays both
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def _get_obs(self):
        # Construct the observation array
        obs = np.zeros(10, dtype=np.int32)
        obs[0] = self.total
        obs[1:] = self.available_numbers
        return obs

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        number = action + 1  # Convert action index to the actual number

        if self.done:
            # If the game is already over
            return self._get_obs(), 0, True, False, {}

        if self.available_numbers[action] == 0:
            # Invalid move: Number not available
            self.done = True
            reward = -10
            return self._get_obs(), reward, True, False, {}

        # Valid move
        self.total += number
        self.available_numbers[action] = 0  # Remove the number from available numbers

        if self.total == 31:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_obs(), reward, True, False, {}

        elif self.total > 31:
            # Current player loses
            self.done = True
            reward = -1
            return self._get_obs(), reward, True, False, {}

        else:
            # Continue the game
            self.current_player *= -1  # Switch player
            reward = 0
            return self._get_obs(), reward, False, False, {}

    def render(self):
        state_str = f"Total: {self.total}\n"
        available_numbers = [
            i + 1 for i, v in enumerate(self.available_numbers) if v == 1
        ]
        state_str += f"Available Numbers: {available_numbers}\n"
        return state_str

    def valid_moves(self):
        return [i for i in range(9) if self.available_numbers[i] == 1]
