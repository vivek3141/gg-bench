import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(18)

        # Observation space:
        # - First 18 entries: availability of numbers (-9 to -1 and 1 to 9)
        # - Next entry: cumulative sum (-81 to 81)
        # - Last entry: current player (-1 or 1)
        low_obs = np.concatenate(
            (
                np.zeros(18, dtype=np.int32),
                np.array([-81], dtype=np.int32),
                np.array([-1], dtype=np.int32),
            )
        )
        high_obs = np.concatenate(
            (
                np.ones(18, dtype=np.int32),
                np.array([81], dtype=np.int32),
                np.array([1], dtype=np.int32),
            )
        )
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, shape=(20,), dtype=np.int32
        )

        # Mapping from action index to number
        self.numbers_list = np.array(
            [-9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.available_numbers = np.ones(
            18, dtype=np.int32
        )  # 1 for available, 0 for not available
        self.cumulative_sum = 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.concatenate(
            (self.available_numbers, [self.cumulative_sum], [self.current_player])
        )
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If game is over, ignore the action and return the current state
            observation = np.concatenate(
                (self.available_numbers, [self.cumulative_sum], [self.current_player])
            )
            return observation, 0, True, False, {}
        if action < 0 or action >= 18 or self.available_numbers[action] == 0:
            # Invalid action
            self.done = True
            observation = np.concatenate(
                (self.available_numbers, [self.cumulative_sum], [self.current_player])
            )
            return (
                observation,
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Valid action
        selected_number = self.numbers_list[action]
        self.cumulative_sum += selected_number
        self.available_numbers[action] = 0  # Remove the number from available numbers

        # Check for victory conditions
        if self.cumulative_sum == 0:
            # Current player wins
            self.done = True
            observation = np.concatenate(
                (self.available_numbers, [self.cumulative_sum], [self.current_player])
            )
            return observation, 1, True, False, {}
        elif np.all(self.available_numbers == 0):
            # All numbers are exhausted, last player wins
            self.done = True
            observation = np.concatenate(
                (self.available_numbers, [self.cumulative_sum], [self.current_player])
            )
            return observation, 1, True, False, {}
        else:
            # Game continues
            self.current_player *= -1  # Switch player
            observation = np.concatenate(
                (self.available_numbers, [self.cumulative_sum], [self.current_player])
            )
            return observation, 0, False, False, {}

    def render(self):
        available_numbers_str = ", ".join(
            [
                str(num)
                for idx, num in enumerate(self.numbers_list)
                if self.available_numbers[idx] == 1
            ]
        )
        state_str = f"Available Numbers: {available_numbers_str}\n"
        state_str += f"Cumulative Sum: {self.cumulative_sum}\n"
        state_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return state_str

    def valid_moves(self):
        return [idx for idx in range(18) if self.available_numbers[idx] == 1]
