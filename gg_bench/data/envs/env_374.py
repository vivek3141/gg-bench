import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 55 possible moves (start and end positions for flipping bits)
        self.action_space = spaces.Discrete(55)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.int32)

        # Prepare action to (start, end) mapping
        self.action_to_move = []
        for start in range(10):
            for end in range(start, 10):
                self.action_to_move.append((start, end))

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bits = np.zeros(10, dtype=np.int32)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.bits.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.bits.copy(), 0, True, False, {}

        if action < 0 or action >= len(self.action_to_move):
            # Invalid action index
            reward = -10
            self.done = True
            return self.bits.copy(), reward, True, False, {}

        start, end = self.action_to_move[action]

        # Check if the bits from start to end are all zeros
        if np.all(self.bits[start : end + 1] == 0):
            # Valid move; flip the bits from start to end to ones
            self.bits[start : end + 1] = 1

            # Check if there are any zeros left
            if np.all(self.bits == 1):
                # Current player wins
                reward = 1
                self.done = True
                return self.bits.copy(), reward, True, False, {}
            else:
                # Continue the game
                reward = 0
                self.current_player = 3 - self.current_player  # Switch player
                return self.bits.copy(), reward, False, False, {}
        else:
            # Invalid move; some bits are already ones
            reward = -10
            self.done = True
            return self.bits.copy(), reward, True, False, {}

    def render(self):
        bit_positions = " ".join(str(i + 1) for i in range(10))
        bit_values = " ".join(str(bit) for bit in self.bits)
        board_str = f"Bit Positions: {bit_positions}\nBit Values:    {bit_values}\n"
        return board_str

    def valid_moves(self):
        valid_actions = []
        for idx, (start, end) in enumerate(self.action_to_move):
            if np.all(self.bits[start : end + 1] == 0):
                valid_actions.append(idx)
        return valid_actions
