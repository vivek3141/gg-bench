import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(
            8
        )  # Actions 0-7 correspond to bit positions 1-8
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bits = np.zeros(8, dtype=np.int8)  # All bits set to '0'
        self.current_player = (
            1  # Player 1 starts; can also represent as -1 for Player 2
        )
        self.done = False
        return self.bits.copy(), {}  # Return observation and info

    def step(self, action):
        # Check if the action is valid
        if not self.action_space.contains(action) or self.done:
            return self.bits.copy(), -10, True, False, {}  # Invalid move

        # Flip the bit at the specified position
        self.bits[action] = 1 - self.bits[action]  # Flip '0' to '1' or '1' to '0'

        # Check for victory condition
        total_ones = np.sum(self.bits)
        if total_ones == 6:
            self.done = True
            reward = 1  # Current player wins
            return self.bits.copy(), reward, True, False, {}

        # Game continues; switch to the other player
        self.current_player *= -1
        reward = 0  # No win yet

        return self.bits.copy(), reward, False, False, {}

    def render(self):
        # Create a visual representation of the bits
        bit_str = " ".join(str(bit) for bit in self.bits)
        return f"Current bits: {bit_str}"

    def valid_moves(self):
        # Return a list of valid moves (bit positions 0-7)
        if self.done:
            return []  # No valid moves if the game is over
        else:
            return list(range(8))
