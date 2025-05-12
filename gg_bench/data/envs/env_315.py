import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=8, target_pattern=None):
        super(CustomEnv, self).__init__()

        self.N = N
        # Define action and observation space
        self.action_space = spaces.Discrete(self.N)

        # Observation space: shared binary number (N bits), target pattern (N bits), current player (1 bit)
        # Total length: 2*N + 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2 * self.N + 1,), dtype=np.int8
        )

        # Initialize the target pattern
        if target_pattern is None:
            # Generate a random target pattern of length N
            self.target_pattern = np.random.randint(2, size=self.N, dtype=np.int8)
        else:
            self.target_pattern = np.array(target_pattern, dtype=np.int8)
            assert self.target_pattern.shape == (
                self.N,
            ), "Target pattern must be of length N"

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_binary_number = np.zeros(self.N, dtype=np.int8)
        self.current_player = 1  # Player 1: 1, Player 2: 0
        self.done = False
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            # Game is already over
            observation = self._get_observation()
            return observation, 0, True, False, {}

        if self.shared_binary_number[action] == 1:
            # Invalid move: bit already set to 1
            self.done = True
            reward = -10
            observation = self._get_observation()
            return observation, reward, True, False, {}

        # Valid move: flip the bit from 0 to 1
        self.shared_binary_number[action] = 1

        # Check if the current player wins
        if np.array_equal(self.shared_binary_number, self.target_pattern):
            reward = 1  # Current player wins
            self.done = True
        else:
            reward = 0
            # Switch to the other player
            self.current_player ^= 1  # Toggle between 1 and 0

        observation = self._get_observation()
        return observation, reward, self.done, False, {}

    def render(self):
        shared_str = "Shared Binary Number: " + " ".join(
            map(str, self.shared_binary_number)
        )
        target_str = "Target Pattern: " + " ".join(map(str, self.target_pattern))
        player_str = (
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}"
        )
        return "\n".join([shared_str, target_str, player_str])

    def valid_moves(self):
        # Return a list of valid moves (bit positions that are 0)
        return [i for i in range(self.N) if self.shared_binary_number[i] == 0]

    def _get_observation(self):
        # Concatenate the shared binary number, target pattern, and current player
        observation = np.concatenate(
            [
                self.shared_binary_number,
                self.target_pattern,
                np.array([self.current_player], dtype=np.int8),
            ]
        )
        return observation
