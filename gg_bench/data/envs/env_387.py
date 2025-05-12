import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Action space: Select a number from 1 to 9 (indices 0 to 8)
        self.action_space = spaces.Discrete(9)

        # Observation space: An array of size 9, each element can be -1, 0, or 1
        # -1: Number is in opponent's set
        #  0: Number is in shared pool
        #  1: Number is in current player's set
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # All numbers are initially in the shared pool
        self.shared_pool = [i + 1 for i in range(9)]  # Numbers 1-9
        self.player_sets = {1: [], -1: []}  # Player 1 and Player -1 (Player 2)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        terminated = False
        truncated = False

        if self.done:
            return self._get_observation(), 0, True, False, {}

        if action < 0 or action >= 9:
            # Invalid action index
            self.done = True
            return self._get_observation(), -10, True, False, {}

        number = action + 1  # Convert action index to actual number (1-9)
        if number not in self.shared_pool:
            # Number already taken
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Valid move
        self.shared_pool.remove(number)
        self.player_sets[self.current_player].append(number)

        # Check for win condition
        if self._check_win(self.player_sets[self.current_player]):
            self.done = True
            return self._get_observation(), 1, True, False, {}
        else:
            # Switch players
            self.current_player *= -1
            return self._get_observation(), 0, False, False, {}

    def render(self):
        output = "Available Numbers: " + " ".join(map(str, self.shared_pool)) + "\n"
        output += f"Player 1's Set: {sorted(self.player_sets[1])}\n"
        output += f"Player 2's Set: {sorted(self.player_sets[-1])}\n"
        return output

    def valid_moves(self):
        # Return indices of available numbers (0-based indices)
        return [num - 1 for num in self.shared_pool]

    def _get_observation(self):
        # Build the observation array
        # -1: Number is in opponent's set
        #  0: Number is in shared pool
        #  1: Number is in current player's set
        observation = np.zeros(9, dtype=np.int8)
        for num in self.shared_pool:
            observation[num - 1] = 0  # Number is in shared pool
        for num in self.player_sets[self.current_player]:
            observation[num - 1] = 1  # Current player's numbers
        for num in self.player_sets[-self.current_player]:
            observation[num - 1] = -1  # Opponent's numbers
        return observation

    def _check_win(self, player_numbers):
        if len(player_numbers) < 3:
            return False
        # Generate all combinations of 3 numbers
        for combo in combinations(player_numbers, 3):
            # Check for arithmetic sequence
            sorted_combo = sorted(combo)
            if (sorted_combo[1] - sorted_combo[0]) == (
                sorted_combo[2] - sorted_combo[1]
            ):
                return True
            # Check for geometric sequence (avoid division by zero)
            if sorted_combo[0] != 0 and sorted_combo[1] != 0:
                if (sorted_combo[1] / sorted_combo[0]) == (
                    sorted_combo[2] / sorted_combo[1]
                ):
                    return True
        return False
