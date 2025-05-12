import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium import logger


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Action space: 18 actions (numbers 1-9 placed at beginning or end)
        self.action_space = spaces.Discrete(18)
        self.max_sequence_length = 19  # Maximum length of the sequence
        # Observation space: sequence plus current player indicator
        self.observation_space = spaces.Box(
            low=-1, high=9, shape=(self.max_sequence_length + 1,), dtype=np.int8
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = []
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        self.done = False
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            logger.warn("Called step() after the game is done.")
            return self._get_observation(), 0, True, False, {}

        if action not in self.valid_moves():
            # Invalid move
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Map action to number and position
        number = (action // 2) + 1  # Numbers 1-9
        position_code = action % 2  # 0 for beginning, 1 for end
        position = "beginning" if position_code == 0 else "end"

        # Place the number
        if position == "beginning":
            self.sequence = [number] + self.sequence
        else:
            self.sequence.append(number)

        # Check for palindrome of length at least 3
        if len(self.sequence) >= 3 and self.sequence == self.sequence[::-1]:
            # Current player wins
            self.done = True
            reward = 1
            return self._get_observation(), reward, True, False, {}

        # Switch to next player
        self.current_player *= -1
        observation = self._get_observation()
        return observation, 0, False, False, {}

    def render(self):
        sequence_str = " ".join(map(str, self.sequence))
        return f"Current Sequence: {sequence_str}"

    def valid_moves(self):
        if self.done:
            return []
        return list(range(18))

    def _get_observation(self):
        obs = np.zeros(self.max_sequence_length + 1, dtype=np.int8)
        seq_length = len(self.sequence)
        if seq_length > 0:
            obs[:seq_length] = self.sequence
        # Include current player in the observation
        obs[-1] = self.current_player
        return obs
