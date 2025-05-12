import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: indices 0-19 correspond to numbers -10 to -1 and +1 to +10
        self.action_space = spaces.Discrete(20)

        # Observation space: cumulative total and availability flags for numbers
        # Observation is a numpy array of shape (21,):
        # - observation[0]: cumulative total (float32)
        # - observation[1:21]: availability flags for each number (1.0 for available, 0.0 for used)
        self.observation_space = spaces.Box(
            low=np.array([-200] + [0] * 20),
            high=np.array([200] + [1] * 20),
            dtype=np.float32,
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_total = 0  # Start with cumulative total zero
        self.available_numbers = np.ones(
            20, dtype=np.float32
        )  # All numbers are initially available
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game is not over
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If game is already over
            return self._get_observation(), 0, True, False, {}

        if action < 0 or action >= 20 or self.available_numbers[action] == 0:
            # Invalid action: selected number is not available
            self.done = True
            return self._get_observation(), -10, True, False, {"invalid_action": True}

        # Map action index to the actual number
        number = self._action_to_number(action)

        # Update cumulative total
        self.cumulative_total += number

        # Remove the selected number from available numbers
        self.available_numbers[action] = 0

        # Check if current player wins by bringing cumulative total to zero
        if self.cumulative_total == 0:
            self.done = True
            reward = 1  # Current player wins
            return self._get_observation(), reward, True, False, {}

        # Check if all numbers have been used
        if not np.any(self.available_numbers):
            # No numbers left, current player loses
            self.done = True
            reward = -1  # Current player loses
            return self._get_observation(), reward, True, False, {}

        # Switch to next player
        self.current_player = 2 if self.current_player == 1 else 1

        # Return observation, reward, terminated, truncated, info
        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Visual representation of the environment state as a string
        number_pool = [
            self._action_to_number(i) if self.available_numbers[i] == 1 else "X"
            for i in range(20)
        ]
        number_pool_str = ", ".join(str(num) for num in number_pool)
        return (
            f"Player {self.current_player}'s turn.\n"
            f"Available Numbers: [{number_pool_str}]\n"
            f"Cumulative Total: {self.cumulative_total}"
        )

    def valid_moves(self):
        # Return list of available action indices
        return [i for i in range(20) if self.available_numbers[i] == 1]

    def _get_observation(self):
        # Return observation: cumulative total and availability flags
        observation = np.concatenate(([self.cumulative_total], self.available_numbers))
        return observation.astype(np.float32)

    def _action_to_number(self, action):
        # Map action index to the actual number
        if 0 <= action <= 9:
            return -10 + action  # Actions 0-9 correspond to numbers -10 to -1
        elif 10 <= action <= 19:
            return action - 9  # Actions 10-19 correspond to numbers +1 to +10
        else:
            raise ValueError("Invalid action index")
