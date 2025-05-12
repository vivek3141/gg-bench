import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: bits positions 3 to 0 (left to right)
        self.action_space = spaces.Discrete(4)
        # Observation space: the state of the 4 bits
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bits = np.zeros(4, dtype=np.int8)  # Bits b3 to b0 initialized to 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.bits.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.bits.copy(), 0, True, False, {}

        # Check for invalid action
        if action < 0 or action >= 4 or self.bits[action] != 0:
            self.done = True
            return self.bits.copy(), -10, True, False, {}  # Invalid move

        # Flip the selected bit from 0 to 1
        self.bits[action] = 1

        # Convert bits to decimal number
        # bits[0] is bit position 3 (MSB), bits[3] is bit position 0 (LSB)
        decimal_value = (
            self.bits[0] * 8 + self.bits[1] * 4 + self.bits[2] * 2 + self.bits[3] * 1
        )

        # Check if decimal_value is divisible by 5
        if decimal_value % 5 == 0 and decimal_value != 0:
            self.done = True
            return self.bits.copy(), 1, True, False, {}  # Current player wins

        # Check if there are any bits left to flip
        if np.all(self.bits == 1):
            # Next player has no valid moves, current player wins
            self.done = True
            return self.bits.copy(), 1, True, False, {}

        # Switch current player
        self.current_player *= -1

        return self.bits.copy(), 0, False, False, {}  # Observation, reward, done, info

    def render(self):
        # Return a visual representation of the environment state as a string
        bits_str = "".join(map(str, self.bits))
        decimal_value = (
            self.bits[0] * 8 + self.bits[1] * 4 + self.bits[2] * 2 + self.bits[3] * 1
        )
        output = "Current Binary Number: {} (Decimal {})\n".format(
            bits_str, decimal_value
        )
        if not self.done:
            output += "Player {}'s turn.\n".format(1 if self.current_player == 1 else 2)
            available_bits = [
                3 - i for i in range(4) if self.bits[i] == 0
            ]  # Positions 3 to 0
            output += "Available bits to flip: {}".format(
                ", ".join(map(str, available_bits))
            )
        else:
            output += "Game over.\n"
        return output

    def valid_moves(self):
        return [i for i in range(4) if self.bits[i] == 0]
