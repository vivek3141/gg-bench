import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 or 1
        self.action_space = spaces.Discrete(2)

        # Observation space: Two binary strings represented as arrays of bits
        # Max length of binary strings
        self.MAX_LEN = 20
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, self.MAX_LEN), dtype=np.int8
        )

        # Initialize the game state
        self.player_strings = [[], []]
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.player_strings = [[], []]  # Reset binary strings
        self.current_player = 0  # Player 1 starts
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Game already over
            return (
                self._get_observation(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        if action not in [0, 1]:
            # Invalid action
            self.done = True
            return self._get_observation(), -10, True, False, {}

        if len(self.player_strings[self.current_player]) >= self.MAX_LEN:
            # Cannot add more bits
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Append the action to the current player's binary string
        self.player_strings[self.current_player].append(action)

        # Check for winning condition: Binary number divisible by 5 (excluding 0)
        binary_str = "".join(
            str(bit) for bit in self.player_strings[self.current_player]
        )
        decimal_value = int(binary_str, 2) if binary_str else 0

        if decimal_value != 0 and decimal_value % 5 == 0:
            # Current player wins
            self.done = True
            reward = 1
            observation = self._get_observation()
            return observation, reward, True, False, {}
        else:
            # Continue the game
            reward = 0
            # Switch to the other player
            self.current_player = 1 - self.current_player
            observation = self._get_observation()
            return observation, reward, False, False, {}

    def render(self):
        output = ""
        for i in range(2):
            binary_str = "".join(str(bit) for bit in self.player_strings[i])
            output += f"Player {i + 1} binary string: {binary_str}\n"
        return output

    def valid_moves(self):
        # Always can choose between '0' and '1' if the game is not over
        if self.done:
            return []
        if len(self.player_strings[self.current_player]) >= self.MAX_LEN:
            return []
        return [0, 1]

    def _get_observation(self):
        # Returns the observation of the current game state
        obs = np.zeros((2, self.MAX_LEN), dtype=np.int8)
        for i in range(2):
            bits = self.player_strings[i]
            obs[i, : len(bits)] = bits
        return obs
