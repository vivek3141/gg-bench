import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Discrete numbers from 1 to 10 (actions 0 to 9 correspond to numbers 1 to 10)
        self.action_space = spaces.Discrete(10)

        # Observation space:
        # Array of 11 elements:
        # - Total sum (0 to 50)
        # - Availability of numbers 1 to 10 (1 if available, 0 if used)
        self.observation_space = spaces.Box(
            low=np.array([0] + [0] * 10),
            high=np.array([50] + [1] * 10),
            dtype=np.int64,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_sum = 0
        self.available_numbers = np.ones(
            10, dtype=np.int64
        )  # Numbers 1 to 10 are available
        self.current_player = 1  # Player 1 starts (Player 1: 1, Player 2: -1)
        self.done = False
        return self._get_obs(), {}  # Observation and info

    def _get_obs(self):
        # Compose observation array
        obs = np.concatenate(([self.total_sum], self.available_numbers))
        return obs

    def step(self, action):
        if self.done:
            # Game is already over
            return self._get_obs(), 0, True, False, {}

        if action < 0 or action >= 10:
            # Invalid action index
            self.done = True
            return self._get_obs(), -10, True, False, {"error": "Invalid action index"}

        if self.available_numbers[action] == 0:
            # Number has already been used
            self.done = True
            return self._get_obs(), -10, True, False, {"error": "Number already used"}

        number = action + 1  # Map action to number (0 -> 1, ..., 9 -> 10)
        self.total_sum += number
        self.available_numbers[action] = 0  # Mark the number as used

        if self.total_sum == 50:
            # Current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {"result": "Reached exactly 50"}
        elif self.total_sum > 50:
            # Current player loses
            self.done = True
            return self._get_obs(), -1, True, False, {"result": "Exceeded 50"}
        elif np.all(self.available_numbers == 0):
            # No numbers left, game ends in a draw
            self.done = True
            return (
                self._get_obs(),
                0,
                True,
                False,
                {"result": "No more numbers, game ends in a draw"},
            )
        else:
            # Valid move, game continues
            self.current_player *= -1  # Switch player
            return (
                self._get_obs(),
                0,
                False,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

    def render(self):
        # Generate a string representation of the game state
        available_numbers = np.where(self.available_numbers == 1)[0] + 1
        used_numbers = np.where(self.available_numbers == 0)[0] + 1
        state_repr = (
            f"Total Sum: {self.total_sum}\n"
            f"Available Numbers: {', '.join(map(str, available_numbers))}\n"
            f"Used Numbers: {', '.join(map(str, used_numbers))}\n"
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        )
        return state_repr

    def valid_moves(self):
        # Return indices of available numbers as valid actions
        return list(np.where(self.available_numbers == 1)[0])
