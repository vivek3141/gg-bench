import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 4 possible swap positions in a sequence of length 5
        self.sequence_length = 5
        self.action_space = spaces.Discrete(self.sequence_length - 1)
        self.observation_space = spaces.Box(
            low=1,
            high=self.sequence_length,
            shape=(self.sequence_length,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the sequence with a random permutation of numbers 1 to 5
        self.sequence = np.random.permutation(np.arange(1, self.sequence_length + 1))
        self.current_player = (
            1  # Player One starts (1 for Player One, -1 for Player Two)
        )
        self.done = False
        return self.sequence.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.sequence.copy(), 0, True, False, {}  # Game already over

        if action < 0 or action >= self.sequence_length - 1:
            # Invalid action: action out of bounds
            self.done = True
            return (
                self.sequence.copy(),
                -10,
                True,
                False,
                {},
            )  # Observation, reward, done, truncated, info

        # Perform the swap
        idx = action
        self.sequence[idx], self.sequence[idx + 1] = (
            self.sequence[idx + 1],
            self.sequence[idx],
        )

        # Check for win condition
        if self.current_player == 1:
            # Player One aims for ascending order
            if np.all(self.sequence[:-1] <= self.sequence[1:]):
                # Sequence is in ascending order
                self.done = True
                return self.sequence.copy(), 1, True, False, {}
        else:
            # Player Two aims for descending order
            if np.all(self.sequence[:-1] >= self.sequence[1:]):
                # Sequence is in descending order
                self.done = True
                return self.sequence.copy(), 1, True, False, {}

        # Switch player
        self.current_player *= -1

        return self.sequence.copy(), 0, False, False, {}

    def render(self):
        sequence_str = " ".join(map(str, self.sequence))
        return sequence_str

    def valid_moves(self):
        # Valid actions are indices from 0 to sequence_length - 2
        return list(range(self.sequence_length - 1))
