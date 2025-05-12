import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: flip bits at positions 1 to 4 (actions 0 to 3)
        self.action_space = spaces.Discrete(4)

        # Observation space: 4 bits, each can be 0 or 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bits = np.zeros(4, dtype=np.int8)  # Initialize bits to 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.bits.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # Can't make a move if the game is over
            return self.bits.copy(), -10, True, False, {}

        if action < 0 or action >= 4 or self.bits[action] == 1:
            # Invalid move: action is out of bounds or bit is already 1
            self.done = True
            return self.bits.copy(), -10, True, False, {}

        # Flip the bit at the chosen position
        self.bits[action] = 1

        # Calculate the decimal equivalent of the binary number
        bits_str = "".join(map(str, self.bits))
        decimal_value = int(bits_str, 2)

        # Check for win condition
        if decimal_value % 5 == 0 and decimal_value != 0:
            # Current player wins
            self.done = True
            return self.bits.copy(), 1, True, False, {}
        else:
            # Valid move, but not a win
            reward = -10
            # Switch to the next player
            self.current_player = 2 if self.current_player == 1 else 1
            return self.bits.copy(), reward, False, False, {}

    def render(self):
        bits_str = " ".join(map(str, self.bits))
        decimal_value = int("".join(map(str, self.bits)), 2)
        render_output = (
            f"Current Binary Number: {bits_str} (Decimal: {decimal_value})\n"
        )
        if not self.done:
            render_output += f"Player {self.current_player}'s turn."
        else:
            render_output += "Game Over."
        return render_output

    def valid_moves(self):
        # Return list of bit positions (0 to 3) that can be flipped
        return [i for i in range(4) if self.bits[i] == 0]
