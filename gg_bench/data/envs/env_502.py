import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions:
        # - 0-9: Add digit 0-9 to the beginning
        # - 10-19: Add digit 0-9 to the end
        self.action_space = spaces.Discrete(20)

        # Observations:
        # - A sequence of up to 15 digits, padded with -1
        self.observation_space = spaces.Box(low=-1, high=9, shape=(15,), dtype=np.int8)

        # Initialize the state
        self.sequence = []  # The shared number sequence
        self.current_player = 1  # 1 or -1 to represent players
        self.done = False  # Indicates if the game is finished

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = []
        self.current_player = 1
        self.done = False
        observation = self._get_observation()
        return observation, {}  # Return initial observation and empty info

    def step(self, action):
        if self.done:
            # If the game is over, any action is invalid
            return self._get_observation(), -10, True, False, {}
        if action not in range(20):
            # Invalid action (not in action space)
            self.done = True
            return self._get_observation(), -10, True, False, {}

        if len(self.sequence) >= 15:
            # No more moves can be made; game should be over
            self.done = True
            return self._get_observation(), -10, True, False, {}

        # Map action to digit and placement
        digit = action % 10  # Digit from 0 to 9
        placement = action // 10  # 0: beginning, 1: end

        # Add the digit to the sequence
        if placement == 0:
            self.sequence.insert(0, digit)
        else:
            self.sequence.append(digit)

        # Check for palindrome of at least 3 digits
        if len(self.sequence) >= 3 and self.sequence == self.sequence[::-1]:
            # Current player wins
            self.done = True
            return self._get_observation(), 1, True, False, {}

        # Check if the sequence has reached maximum length
        if len(self.sequence) >= 15:
            # Opponent wins
            self.done = True
            return self._get_observation(), -1, True, False, {}

        # Switch players
        self.current_player *= -1

        # Game continues
        return self._get_observation(), 0, False, False, {}

    def render(self):
        sequence_str = "".join(map(str, self.sequence)) if self.sequence else "Empty"
        return f"Current Sequence: {sequence_str}"

    def valid_moves(self):
        if self.done or len(self.sequence) >= 15:
            return []
        return list(range(20))  # All actions are valid until the sequence is full

    def _get_observation(self):
        # Pad the sequence with -1 to maintain a fixed observation size
        observation = np.full(15, -1, dtype=np.int8)
        observation[: len(self.sequence)] = self.sequence
        return observation
