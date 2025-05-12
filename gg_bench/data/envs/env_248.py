import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target=72):
        super(CustomEnv, self).__init__()

        # Set the target number for the game
        self.target = target

        # Define the action space: integers from 0 to 7 corresponding to multipliers 2 to 9
        self.action_space = spaces.Discrete(8)

        # Define the observation space: current cumulative product (integer between 1 and target)
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([self.target]), shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the cumulative product and current player
        self.cumulative_product = 1
        self.current_player = 1  # Player 1 starts; can be 1 or 2
        self.done = False  # Indicates if the game has ended

        # Return the initial observation and an empty info dictionary
        return np.array([self.cumulative_product], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # If the game is over, return the current state with a penalty
            return (
                np.array([self.cumulative_product], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Validate the action
        if action < 0 or action >= 8:
            # Invalid action; end the game with a penalty
            self.done = True
            return (
                np.array([self.cumulative_product], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Map action index to multiplier (2-9)
        multiplier = action + 2

        # Calculate the new cumulative product
        new_product = self.cumulative_product * multiplier

        if new_product > self.target:
            # The cumulative product exceeds the target; current player loses
            self.cumulative_product = new_product
            self.done = True
            reward = -1  # Negative reward for losing
            return (
                np.array([self.cumulative_product], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        elif new_product == self.target:
            # The cumulative product equals the target; current player wins
            self.cumulative_product = new_product
            self.done = True
            reward = 1  # Positive reward for winning
            return (
                np.array([self.cumulative_product], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Valid move; game continues
            self.cumulative_product = new_product
            self.current_player = 3 - self.current_player  # Switch player (1 <-> 2)
            reward = 0  # No reward for a regular move
            return (
                np.array([self.cumulative_product], dtype=np.int32),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        # Return a string representation of the current game state
        return f"Current cumulative product: {self.cumulative_product}"

    def valid_moves(self):
        # Return a list of valid action indices based on the current cumulative product
        valid_actions = []
        for action in range(8):
            multiplier = action + 2
            if self.cumulative_product * multiplier <= self.target:
                valid_actions.append(action)
        return valid_actions
