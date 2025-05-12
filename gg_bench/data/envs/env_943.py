import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the maximum sequence length
        self.max_length = 100

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Actions: 0=>1, 1=>2, 2=>3
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(self.max_length,), dtype=np.int32
        )

        # Initialize the sequence
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = np.zeros(self.max_length, dtype=np.int32)
        self.current_length = 0  # Number of elements in the sequence
        self.current_player = 1  # Players: 1 and 2
        self.done = False
        return self.sequence.copy(), {}

    def step(self, action):
        if self.done:
            return self.sequence.copy(), -10, True, False, {}

        if action not in [0, 1, 2]:
            return self.sequence.copy(), -10, True, False, {}

        # Map action to number between 1 and 3
        number = action + 1  # action 0=>1, 1=>2, 2=>3

        # Append number to sequence
        if self.current_length >= self.max_length:
            self.done = True
            return self.sequence.copy(), -10, True, False, {}

        self.sequence[self.current_length] = number
        self.current_length += 1

        # Check for win/loss conditions
        if self.current_length >= 3:
            sum_last_three = np.sum(
                self.sequence[self.current_length - 3 : self.current_length]
            )
            if sum_last_three == 6:
                self.done = True
                return self.sequence.copy(), 1, True, False, {}
            elif sum_last_three > 6:
                self.done = True
                return self.sequence.copy(), -10, True, False, {}

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

        return self.sequence.copy(), 0, False, False, {}

    def render(self):
        seq_str = (
            "Sequence: ["
            + ", ".join(map(str, self.sequence[: self.current_length]))
            + "]"
        )
        return seq_str

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1, 2]  # All actions are valid when the game is ongoing
