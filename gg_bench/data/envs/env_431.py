import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0-14
        # Actions 0-7: Flip bit 1-8
        # Actions 8-14: Flip bits (1&2)-(7&8)
        self.action_space = spaces.Discrete(15)

        # Observation space: 8 bits, 0 or 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.int8)

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

        # Check if current player has any valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Current player cannot move, they lose
            self.done = True
            return self.bits.copy(), -1, True, False, {}

        # Check if action is valid
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self.bits.copy(), -10, True, False, {}

        # Apply action
        self._apply_action(action)

        # Check if current player wins (all bits set to 1)
        if np.all(self.bits == 1):
            self.done = True
            return self.bits.copy(), 1, True, False, {}

        # Switch to next player
        self.current_player *= -1

        # Check if next player has any valid moves
        valid_actions_next = self.valid_moves()
        if not valid_actions_next:
            # Next player cannot move, current player wins
            self.done = True
            # Switch back to current player to report reward correctly
            self.current_player *= -1
            return self.bits.copy(), 1, True, False, {}

        # Game continues
        return self.bits.copy(), 0, False, False, {}

    def render(self):
        positions = " ".join(f"[{i+1}]" for i in range(8))
        bits_display = " ".join(f"[{b}]" for b in self.bits)
        board_str = f"Positions: {positions}\nBits:      {bits_display}\n"
        print(board_str)

    def valid_moves(self):
        valid_actions = []

        # Single bit flips (actions 0-7)
        for i in range(8):
            if self.bits[i] == 0:
                valid_actions.append(i)

        # Two adjacent bit flips (actions 8-14)
        for i in range(7):
            if self.bits[i] == 0 and self.bits[i + 1] == 0:
                valid_actions.append(8 + i)

        return valid_actions

    def _apply_action(self, action):
        if 0 <= action <= 7:
            # Flip one bit
            if self.bits[action] == 0:
                self.bits[action] = 1
        elif 8 <= action <= 14:
            # Flip two adjacent bits
            idx = action - 8
            if self.bits[idx] == 0 and self.bits[idx + 1] == 0:
                self.bits[idx] = 1
                self.bits[idx + 1] = 1
