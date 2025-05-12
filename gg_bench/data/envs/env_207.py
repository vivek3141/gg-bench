import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_divisor=7):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # The action space is Discrete(8), representing bit positions 0-7 (bits 1-8)
        self.action_space = spaces.Discrete(8)

        # The observation space is an array of 8 bits (0 or 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.int8)

        # Set the target divisor
        self.target_divisor = target_divisor

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bits = np.zeros(8, dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.bits.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.bits.copy(), 0, True, False, {}
        if self.bits[action] == 1:
            # Invalid move
            self.done = True
            return self.bits.copy(), -10, True, False, {}
        else:
            # Valid move
            self.bits[action] = 1  # Flip the bit from 0 to 1
            # Convert binary bits to decimal number
            binary_str = "".join(map(str, self.bits))
            decimal_number = int(binary_str, 2)

            # Check for win condition
            if decimal_number % self.target_divisor == 0 and decimal_number != 0:
                self.done = True
                return self.bits.copy(), 1, True, False, {}
            else:
                # Check for available moves
                if np.all(self.bits == 1):
                    # No valid moves left, opponent wins
                    self.done = True
                    return self.bits.copy(), -1, True, False, {}
                else:
                    # Switch player
                    self.current_player = 2 if self.current_player == 1 else 1
                    return self.bits.copy(), 0, False, False, {}

    def render(self):
        binary_str = "".join(map(str, self.bits))
        decimal_number = int(binary_str, 2)
        render_str = f"Current Binary Number: {binary_str} (Decimal {decimal_number})\n"
        render_str += f"Current Player: Player {self.current_player}"
        return render_str

    def valid_moves(self):
        return [i for i in range(8) if self.bits[i] == 0]
