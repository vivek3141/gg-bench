import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the mapping from indices to numbers
        self.number_mapping = np.array(
            [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            dtype=np.int32,
        )

        # Action space: indices from 0 to 19
        self.action_space = spaces.Discrete(20)

        # Observation space: 22 elements
        # First 20 elements: availability of numbers (1 or 0)
        # 21st element: cumulative sum (-55 to 55)
        # 22nd element: current player (-1 or 1)
        self.observation_space = spaces.Box(
            low=np.array([0] * 20 + [-55, -1], dtype=np.float32),
            high=np.array([1] * 20 + [55, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # Initialize the game
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers are available at the start
        self.available_numbers = np.ones(20, dtype=np.float32)
        self.cumulative_sum = 0
        self.current_player = 1
        self.done = False
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # action is an integer from 0 to 19
        if self.done:
            # Game is already over
            observation = self._get_obs()
            reward = -10
            terminated = True
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info

        if action < 0 or action >= 20 or self.available_numbers[action] == 0:
            # Invalid action
            observation = self._get_obs()
            reward = -10
            terminated = True
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info

        # Valid action
        selected_number = self.number_mapping[action]
        self.available_numbers[action] = 0  # Remove number from availability
        self.cumulative_sum += selected_number

        # Check for win
        if self.cumulative_sum == 0:
            # Current player wins
            self.done = True
            observation = self._get_obs()
            reward = 1
            terminated = True
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info

        # Check if all numbers are exhausted
        if np.sum(self.available_numbers) == 0:
            # All numbers are exhausted and cumulative sum is not zero
            # Current player loses
            self.done = True
            observation = self._get_obs()
            reward = -1
            terminated = True
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info

        # Game continues
        # Switch current player
        self.current_player *= -1
        observation = self._get_obs()
        reward = -10  # As per the prompt
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        s = f"Current Cumulative Sum: {self.cumulative_sum}\n"
        s += "Available Numbers:\n"
        available_numbers_values = self.number_mapping[self.available_numbers == 1]
        s += ", ".join(map(str, available_numbers_values)) + "\n"
        s += f"Player {1 if self.current_player == 1 else 2}'s turn\n"
        return s

    def valid_moves(self):
        # Return indices of available numbers
        return [i for i in range(20) if self.available_numbers[i] == 1]

    def _get_obs(self):
        # Observation is an array of size 22
        # First 20 elements: available numbers (1 or 0)
        # 21st element: cumulative sum
        # 22nd element: current player (-1 or 1)
        obs = np.concatenate(
            (self.available_numbers, [self.cumulative_sum], [self.current_player])
        )
        return obs.astype(np.float32)
