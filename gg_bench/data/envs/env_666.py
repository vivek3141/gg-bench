import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define the allowed prime multipliers
        self.prime_multipliers = [2, 3, 5, 7]

        # Define action and observation space
        # Actions are indices of the prime multipliers list
        self.action_space = spaces.Discrete(len(self.prime_multipliers))
        # Observation is the current shared total
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([1000]), dtype=np.int32
        )

        # Initialize the game state
        self.current_player = 1  # Player 1 starts
        self.shared_total = 1
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_total = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.shared_total], dtype=np.int32),
            {},
        )  # Return observation and info

    def step(self, action):
        # Check if game is already over
        if self.done:
            return (
                np.array([self.shared_total], dtype=np.int32),
                0,
                self.done,
                False,
                {},
            )

        # Check if current player has any valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # No valid moves, current player loses
            self.done = True
            reward = -10  # Current player loses
            return (
                np.array([self.shared_total], dtype=np.int32),
                reward,
                self.done,
                False,
                {},
            )

        # Check if the action is valid
        if action not in valid_actions:
            # Invalid move, current player loses
            self.done = True
            reward = -10  # Current player loses
            return (
                np.array([self.shared_total], dtype=np.int32),
                reward,
                self.done,
                False,
                {},
            )

        # Apply the action
        multiplier = self.prime_multipliers[action]
        self.shared_total *= multiplier

        # Check for win
        if self.shared_total == 1000:
            self.done = True
            reward = 1  # Current player wins
            return (
                np.array([self.shared_total], dtype=np.int32),
                reward,
                self.done,
                False,
                {},
            )

        # Check for loss (exceeded 1000)
        elif self.shared_total > 1000:
            self.done = True
            reward = -10  # Current player loses
            return (
                np.array([self.shared_total], dtype=np.int32),
                reward,
                self.done,
                False,
                {},
            )

        # Game continues
        else:
            # Switch player
            self.current_player *= -1  # Swap between 1 and -1
            reward = 0  # No reward for a regular move
            return (
                np.array([self.shared_total], dtype=np.int32),
                reward,
                self.done,
                False,
                {},
            )

    def render(self):
        return f"Current Shared Total: {self.shared_total}"

    def valid_moves(self):
        valid_actions = []
        # Check each action to see if it results in a total <= 1000
        for idx, multiplier in enumerate(self.prime_multipliers):
            potential_total = self.shared_total * multiplier
            if potential_total <= 1000:
                valid_actions.append(idx)
        return valid_actions
