import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.max_number = 100  # Maximum starting number and maximum possible number

        # Actions correspond to selecting numbers from 1 to max_number inclusive
        # So action indices from 0 to max_number -1 correspond to numbers 1 to max_number
        self.action_space = spaces.Discrete(self.max_number)

        # Observation space is a vector of length max_number + 1
        # observation[0] = current_number normalized between 0 and 1
        # observation[1:] = indicators for numbers from 1 to max_number, 1 if used, 0 otherwise
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.max_number + 1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_number = self.max_number
        self.used_numbers = set()
        self.current_player = 1  # Start with Player 1
        self.done = False

        observation = self._get_observation()
        return observation, {}  # Return observation and info dict

    def step(self, action):
        if self.done:
            # Game is over
            return self._get_observation(), 0.0, True, False, {}

        selected_number = action + 1  # Map action index to selected number

        # Validate action
        valid = self._is_valid_action(selected_number)

        if not valid:
            # Invalid action, player loses
            self.done = True
            reward = -10.0  # Penalty for invalid move
            return self._get_observation(), reward, True, False, {}

        # Valid action
        self.current_number -= selected_number
        self.used_numbers.add(selected_number)

        if self.current_number == 0:
            # Current player wins
            self.done = True
            reward = 1.0
            return self._get_observation(), reward, True, False, {}

        # Swap player
        self.current_player = 3 - self.current_player  # Swaps between 1 and 2

        # Check if next player has any valid moves
        if not self.valid_moves():
            # Next player cannot make a valid move, current player wins
            self.done = True
            reward = 1.0  # Current player wins
            return self._get_observation(), reward, True, False, {}

        # Game continues
        reward = 0.0
        return self._get_observation(), reward, False, False, {}

    def render(self):
        info = f"Current Number: {self.current_number}\n"
        info += f"Used Numbers: {sorted(self.used_numbers)}\n"
        info += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        return info

    def valid_moves(self):
        valid_moves = []
        for num in range(1, self.max_number + 1):
            if self.current_number % num == 0 and num not in self.used_numbers:
                valid_moves.append(
                    num - 1
                )  # action indices are from 0 to max_number -1
        return valid_moves

    def _get_observation(self):
        obs = np.zeros(self.max_number + 1, dtype=np.float32)
        obs[0] = self.current_number / self.max_number
        for num in self.used_numbers:
            obs[num] = 1.0
        return obs

    def _is_valid_action(self, selected_number):
        # Check if selected_number is a factor of current_number
        if selected_number < 1 or selected_number > self.max_number:
            return False
        if self.current_number % selected_number != 0:
            return False
        # Check if selected_number has not been used
        if selected_number in self.used_numbers:
            return False
        return True
