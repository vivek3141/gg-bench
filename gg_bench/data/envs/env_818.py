import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(15)  # Actions from 0 to 14
        self.observation_space = spaces.Box(low=0, high=15, shape=(16,), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = list(range(1, 16))
        self.phase = 0  # 0 for partitioning, 1 for selecting
        self.done = False
        self.current_player = 1  # Can be 1 or -1, but not used as it's self-play
        self.segment1 = []
        self.segment2 = []
        observation = self._get_observation()
        return observation, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, False, {}

        if self.phase == 0:
            # Partition phase
            valid_actions = self.valid_moves()
            if action not in valid_actions:
                # Invalid action
                self.done = True
                return self._get_observation(), -10, True, False, {}
            split_pos = action
            self.segment1 = self.sequence[:split_pos]
            self.segment2 = self.sequence[split_pos:]
            self.phase = 1  # Switch to selection phase
            reward = 0
            terminated = False
        else:
            # Selection phase
            if action not in [0, 1]:
                # Invalid action
                self.done = True
                return self._get_observation(), -10, True, False, {}
            if action == 0:
                # Remove segment 1
                self.sequence = self.segment2
            else:
                # Remove segment 2
                self.sequence = self.segment1
            self.phase = 0  # Switch back to partition phase
            self.segment1 = []
            self.segment2 = []
            if len(self.sequence) == 1:
                # Current player loses
                reward = -1
                self.done = True
                terminated = True
            else:
                reward = 0
                terminated = False
        observation = self._get_observation()
        return observation, reward, terminated, False, {}

    def render(self):
        sequence_str = ", ".join(map(str, self.sequence))
        phase_str = "Partitioning" if self.phase == 0 else "Selecting"
        return f"Current Sequence: [{sequence_str}]\nPhase: {phase_str}"

    def valid_moves(self):
        if self.phase == 0:
            # Partition phase: valid split positions
            return list(range(1, len(self.sequence)))
        else:
            # Selection phase: choose between segment 0 or 1
            return [0, 1]

    def _get_observation(self):
        obs = np.zeros(16, dtype=np.int32)
        seq_len = len(self.sequence)
        obs[:seq_len] = self.sequence
        obs[15] = self.phase  # Phase indicator
        return obs
