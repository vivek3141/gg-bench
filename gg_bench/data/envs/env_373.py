import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # There are 15 possible actions:
        # Actions 0-7: Single Flip at positions 1-8
        # Actions 8-14: Double Flip with Risk at positions 1-7 (flipping bits i and i+1)
        self.action_space = spaces.Discrete(15)

        # Observation space consists of both players' bytes (16 bits total)
        # Each bit can be 0 or 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_bytes = np.zeros((2, 8), dtype=np.int8)
        self.current_player = 0  # 0 for Player 1, 1 for Player 2
        self.done = False
        return self._get_observation(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Perform action
        if 0 <= action <= 7:
            # Single Flip
            bit_pos = action
            self.player_bytes[self.current_player][bit_pos] = 1
        elif 8 <= action <= 14:
            # Double Flip with Risk
            first_bit = action - 8
            second_bit = first_bit + 1
            bits = self.player_bytes[self.current_player]
            bits[first_bit] = 1
            bits[second_bit] = 1

            # Risk: revert one random other '1' bit back to '0'
            one_indices = np.where(bits == 1)[0]
            # Exclude the two bits just flipped
            one_indices = one_indices[
                (one_indices != first_bit) & (one_indices != second_bit)
            ]
            if len(one_indices) > 0:
                # Randomly choose one to revert
                bit_to_revert = self.np_random.choice(one_indices)
                bits[bit_to_revert] = 0
        else:
            # Invalid action id
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Check for win
        if np.all(self.player_bytes[self.current_player] == 1):
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Switch player
        self.current_player = 1 - self.current_player

        return self._get_observation(), 0, False, False, {}

    def render(self):
        # Return a string representation of the game state
        s = ""
        for i in range(2):
            s += f'Player {i+1} bits: {" ".join(map(str, self.player_bytes[i]))}\n'
        s += f"Current player: Player {self.current_player + 1}\n"
        return s

    def valid_moves(self):
        valid_actions = []
        bits = self.player_bytes[self.current_player]

        # Single Flip actions
        zero_indices = np.where(bits == 0)[0]
        for i in zero_indices:
            valid_actions.append(i)

        # Double Flip actions
        for i in range(7):
            if bits[i] == 0 and bits[i + 1] == 0:
                valid_actions.append(i + 8)  # Actions 8-14

        return valid_actions

    def _get_observation(self):
        return np.concatenate(self.player_bytes)
