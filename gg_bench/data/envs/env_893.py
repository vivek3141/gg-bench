import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(9), representing numbers 1-9
        self.action_space = spaces.Discrete(9)

        # The observation space includes:
        # - Central pool availability: 9 elements (0 or 1)
        # - Player sums: 2 elements (player 1 sum, player 2 sum)
        self.observation_space = spaces.Box(
            low=np.zeros(11, dtype=np.int32),
            high=np.array([1] * 9 + [45, 45], dtype=np.int32),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.central_pool = np.ones(9, dtype=np.int32)  # Numbers 1-9 are available
        self.player_sums = np.array([0, 0], dtype=np.int32)  # Player sums start at 0
        self.current_player = 0  # Player 1 starts (index 0)
        self.terminated = False  # Game is not over

        return self._get_observation(), {}

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, {}

        # Validate the action
        if action < 0 or action >= 9 or self.central_pool[action] == 0:
            # Invalid move
            reward = -10
            self.terminated = True
            return self._get_observation(), reward, True, False, {}

        number = action + 1  # Map action to number (0 -> 1, ..., 8 -> 9)

        # Update the player's sum and remove the number from the pool
        self.player_sums[self.current_player] += number
        self.central_pool[action] = 0

        # Check if the current player's sum is a multiple of five
        if self.player_sums[self.current_player] % 5 == 0:
            reward = -10
            self.terminated = True
            return self._get_observation(), reward, True, False, {}

        # Check if the central pool is empty
        if np.sum(self.central_pool) == 0:
            # Current player wins if no numbers remain
            reward = 1
            self.terminated = True
            return self._get_observation(), reward, True, False, {}

        # Switch to the other player
        self.current_player = 1 - self.current_player

        return self._get_observation(), 0, False, False, {}

    def render(self):
        central_numbers = [str(i + 1) for i in range(9) if self.central_pool[i] == 1]
        s = f"Numbers available: {' '.join(central_numbers)}\n"
        s += f"Player 1 Sum: {self.player_sums[0]}\n"
        s += f"Player 2 Sum: {self.player_sums[1]}\n"
        s += f"Player {self.current_player + 1}'s turn."
        return s

    def valid_moves(self):
        return [i for i in range(9) if self.central_pool[i] == 1]

    def _get_observation(self):
        # Combine central pool and player sums into a single observation array
        return np.concatenate([self.central_pool, self.player_sums])
