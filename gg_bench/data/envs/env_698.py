import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 7 possible actions:
        # 0: flip bit 1
        # 1: flip bit 2
        # 2: flip bit 3
        # 3: flip bit 4
        # 4: flip bits 1 and 2
        # 5: flip bits 2 and 3
        # 6: flip bits 3 and 4
        self.action_space = spaces.Discrete(7)

        # Observation space: 4 bits, values 0 or 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8)

        self.bits = None
        self.current_player = None
        self.done = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bits = np.array([0, 0, 0, 0], dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.bits.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, return current state
            return self.bits.copy(), 0, True, False, {}

        if action not in range(7):
            # Invalid action
            reward = -10
            self.done = True
            return self.bits.copy(), reward, True, False, {}

        # Perform the action
        if action == 0:
            self.bits[0] ^= 1  # Flip bit 1
        elif action == 1:
            self.bits[1] ^= 1  # Flip bit 2
        elif action == 2:
            self.bits[2] ^= 1  # Flip bit 3
        elif action == 3:
            self.bits[3] ^= 1  # Flip bit 4
        elif action == 4:
            self.bits[0] ^= 1  # Flip bits 1 and 2
            self.bits[1] ^= 1
        elif action == 5:
            self.bits[1] ^= 1  # Flip bits 2 and 3
            self.bits[2] ^= 1
        elif action == 6:
            self.bits[2] ^= 1  # Flip bits 3 and 4
            self.bits[3] ^= 1

        # Check for win
        if np.all(self.bits == 1):
            reward = 1
            self.done = True
            return self.bits.copy(), reward, True, False, {}

        # Switch current player
        self.current_player *= -1

        return self.bits.copy(), 0, False, False, {}

    def render(self):
        bits_str = " ".join(map(str, self.bits))
        player_str = "Player 1" if self.current_player == 1 else "Player 2"
        return f"Current State: {bits_str}\n{player_str}'s turn."

    def valid_moves(self):
        # All actions from 0 to 6 are valid in any state
        return list(range(7))
