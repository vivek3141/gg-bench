import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Append 0 or 1 to the sequence
        self.action_space = spaces.Discrete(2)

        # Observation space: [remainder_mod_3, current_player]
        self.observation_space = spaces.MultiDiscrete([3, 2])

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = []  # Binary sequence as a list of bits
        self.remainder = 0  # Current remainder modulo 3
        self.current_player = 0  # Player 0 starts
        self.done = False
        return np.array([self.remainder, self.current_player], dtype=np.int32), {}

    def step(self, action):
        # Ensure the action is valid
        if action not in [0, 1]:
            reward = -10
            self.done = True
            return (
                np.array([self.remainder, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Append the action to the sequence
        self.sequence.append(action)

        # Update the remainder modulo 3
        self.remainder = (2 * self.remainder + action) % 3

        # Check for win condition
        if self.remainder == 0 and len(self.sequence) > 0:
            reward = 1  # Current player wins
            self.done = True
            return (
                np.array([self.remainder, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            reward = 0  # No reward for ongoing play

        # Switch to the next player
        self.current_player = 1 - self.current_player

        return (
            np.array([self.remainder, self.current_player], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        # Reconstruct the binary sequence as a string
        sequence_str = "".join(str(bit) for bit in self.sequence)
        decimal_value = int(sequence_str, 2) if sequence_str else 0
        output = f"Current Sequence: {sequence_str or '(empty)'}\n"
        output += f"Decimal Value: {decimal_value}\n"
        output += f"Current Player: Player {self.current_player}\n"
        print(output)

    def valid_moves(self):
        # Both 0 and 1 are always valid moves
        return [0, 1]
