import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Starting number is 31 (binary 11111)
        self.starting_number = 31
        self.num_bits = self.starting_number.bit_length()

        # Define action and observation space
        self.action_space = spaces.Discrete(self.num_bits)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_bits,), dtype=np.int8
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize bits array with the binary representation of the starting number
        self.bits = np.array(
            [int(bit) for bit in bin(self.starting_number)[2:].zfill(self.num_bits)],
            dtype=np.int8,
        )
        self.current_player = 1
        self.done = False
        return self.bits.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.bits.copy(), 0, True, False, {}

        # Validate action
        if action < 0 or action >= self.num_bits or self.bits[action] == 0:
            self.done = True
            return self.bits.copy(), -10, True, False, {}

        # Flip the selected '1' bit to '0'
        self.bits[action] = 0

        # Check if the current player wins
        if np.sum(self.bits) == 0:
            self.done = True
            return self.bits.copy(), 1, True, False, {}

        # Switch to the other player
        self.current_player *= -1
        return self.bits.copy(), 0, False, False, {}

    def render(self):
        binary_str = "".join(str(bit) for bit in self.bits)
        decimal_value = int(binary_str, 2)
        bit_positions = " ".join(f"{i+1}" for i in range(self.num_bits))
        bits_with_positions = " ".join(f"{bit}" for bit in self.bits)
        return (
            f"Current Player: Player {1 if self.current_player == 1 else 2}\n"
            f"Current Number: {decimal_value} (binary {binary_str})\n"
            f"Bit Positions: {bit_positions}\n"
            f"Bits:          {bits_with_positions}"
        )

    def valid_moves(self):
        return [i for i in range(self.num_bits) if self.bits[i] == 1]
