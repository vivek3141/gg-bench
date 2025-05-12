import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Discrete(8)  # Bits positions from 0 to 7
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(16,), dtype=np.int8
        )  # Bits of both players

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bits = np.ones((2, 8), dtype=np.int8)  # [player][bits]
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        opponent = 1 - self.current_player
        # Check if action is valid: opponent's bit at position 'action' must be 1
        if self.bits[opponent][action] == 0:
            # Invalid move
            self.done = True
            return self._get_obs(), -10, True, False, {}

        # Flip the opponent's bit at 'action'
        self.bits[opponent][action] = 0

        # Check if opponent's bits sum to zero (opponent has no bits set to 1)
        if np.sum(self.bits[opponent]) == 0:
            # Current player wins
            self.done = True
            return self._get_obs(), 1, True, False, {}

        # Switch current player
        self.current_player = opponent

        return self._get_obs(), 0, False, False, {}

    def render(self):
        player1_bits = "".join(map(str, self.bits[0]))
        player2_bits = "".join(map(str, self.bits[1]))
        player1_decimal = int("".join(map(str, self.bits[0])), 2)
        player2_decimal = int("".join(map(str, self.bits[1])), 2)
        render_str = (
            f"Player 1's Number: {player1_decimal} (binary {player1_bits})\n"
            f"Player 2's Number: {player2_decimal} (binary {player2_bits})\n"
        )
        return render_str

    def valid_moves(self):
        opponent = 1 - self.current_player
        return [i for i in range(8) if self.bits[opponent][i] == 1]

    def _get_obs(self):
        # Combine the bits of both players into a single observation
        return np.concatenate((self.bits[0], self.bits[1]))
