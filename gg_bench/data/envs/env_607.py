import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: Integers from 0 to 9999, representing all possible subranges
        self.action_space = spaces.Discrete(10000)

        # Observation space: The current lower and upper bounds of the range
        self.observation_space = spaces.Box(
            low=np.array([1, 1]), high=np.array([100, 100]), shape=(2,), dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.lower_bound = 1
        self.upper_bound = 100
        self.current_player = 1  # Player 1 starts
        self.done = False

        observation = np.array([self.lower_bound, self.upper_bound], dtype=np.int32)
        return observation, {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.lower_bound, self.upper_bound], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Decode action into new_lower and new_upper
        new_lower = action // 100 + 1
        new_upper = action % 100 + 1

        # Check if action is valid
        if not self.is_valid_action(new_lower, new_upper):
            reward = -10
            self.done = True
            return (
                np.array([self.lower_bound, self.upper_bound], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Apply the action
        self.lower_bound = new_lower
        self.upper_bound = new_upper

        # Check if the next player has any valid moves
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            # Current player wins
            reward = 1
            self.done = True
        else:
            # Game continues
            reward = -10
            self.done = False
            self.current_player = 3 - self.current_player  # Switch player

        observation = np.array([self.lower_bound, self.upper_bound], dtype=np.int32)
        return (
            observation,
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        return f"Current player: Player {self.current_player}, Current range: {self.lower_bound}-{self.upper_bound}"

    def valid_moves(self):
        moves = []
        if self.upper_bound - self.lower_bound <= 2:
            return moves  # No valid moves available
        # Generate all valid subranges excluding at least one number from both ends
        for new_lower in range(self.lower_bound + 1, self.upper_bound - 1):
            for new_upper in range(new_lower, self.upper_bound - 1 + 1):
                action = (new_lower - 1) * 100 + (new_upper - 1)
                moves.append(action)
        return moves

    def is_valid_action(self, new_lower, new_upper):
        # Action must exclude at least one number from both ends
        if new_lower <= self.lower_bound or new_upper >= self.upper_bound:
            return False
        # Subrange must be valid
        if new_lower > new_upper:
            return False
        return True
