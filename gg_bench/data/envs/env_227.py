import numpy as np
import gymnasium as gym
from gymnasium import spaces
from itertools import combinations


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Total possible flip combinations (choosing 1 to 3 bits from positions 1 to 8)
        # Generate all combinations of positions (0 to 7) for flipping 1 to 3 bits
        self.action_list = []
        for k in [1, 2, 3]:
            self.action_list.extend(list(combinations(range(8), k)))
        self.num_actions = len(self.action_list)  # Should be 92 actions
        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space: 24 bits (current player's byte, opponent's byte, target byte)
        self.observation_space = spaces.Box(low=0, high=1, shape=(24,), dtype=np.int8)

        self.seed()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize players' bytes to zeros
        self.player_bytes = [np.zeros(8, dtype=np.int8), np.zeros(8, dtype=np.int8)]
        # Randomly generate the target byte
        self.target_byte = self.np_random.integers(0, 2, size=8, dtype=np.int8)
        # Set current player (0 or 1)
        self.current_player = 0
        self.done = False
        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        # Create an observation combining current player's byte, opponent's byte, and target byte
        current_byte = self.player_bytes[self.current_player]
        opponent_byte = self.player_bytes[1 - self.current_player]
        observation = np.concatenate((current_byte, opponent_byte, self.target_byte))
        return observation

    def step(self, action):
        # Check if the game is already over
        if self.done:
            reward = -10
            return self._get_observation(), reward, True, False, {}
        # Validate the action
        if action < 0 or action >= self.num_actions:
            reward = -10
            self.done = True
            return self._get_observation(), reward, True, False, {}
        # Get the positions to flip based on the action
        flip_positions = self.action_list[action]
        # Flip the bits in the current player's byte
        player_byte = self.player_bytes[self.current_player].copy()
        player_byte[list(flip_positions)] ^= 1  # Flip bits using XOR
        self.player_bytes[self.current_player] = player_byte
        # Check for victory condition
        if np.array_equal(player_byte, self.target_byte):
            reward = 1
            self.done = True
            return self._get_observation(), reward, True, False, {}
        else:
            reward = -10
            self.current_player = 1 - self.current_player  # Switch to the other player
            return self._get_observation(), reward, False, False, {}

    def render(self):
        # Generate a string representation of the current game state
        current_byte = self.player_bytes[self.current_player]
        opponent_byte = self.player_bytes[1 - self.current_player]
        target_byte = self.target_byte
        s = f"Player {self.current_player + 1}'s turn\n"
        s += f"Your byte:       {''.join(map(str, current_byte))}\n"
        s += f"Opponent's byte: {''.join(map(str, opponent_byte))}\n"
        s += f"Target byte:     {''.join(map(str, target_byte))}\n"
        return s

    def valid_moves(self):
        # Return a list of valid action indices
        if self.done:
            return []
        else:
            return list(range(self.num_actions))
