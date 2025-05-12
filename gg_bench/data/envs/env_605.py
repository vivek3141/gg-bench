import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions are integers from 0 to 9, corresponding to numbers 1 to 10
        self.action_space = spaces.Discrete(10)
        # Observation is the cumulative sum as an integer from 0 to 50 inclusive
        self.observation_space = spaces.Box(
            low=np.array([0]), high=np.array([50]), shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_sum = 0
        self.current_player = 1  # Player 1 starts the game
        self.done = False
        return (
            np.array([self.cumulative_sum], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            # If the game is over, return the current state
            return (
                np.array([self.cumulative_sum], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Map action (0-9) to number added (1-10)
        number_added = action + 1

        # Update cumulative sum
        self.cumulative_sum += number_added

        # Check for losing condition (sum exceeds 50)
        if self.cumulative_sum > 50:
            reward = -10  # Player loses
            self.done = True
            return (
                np.array([self.cumulative_sum], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check for winning condition (sum equals 50)
        if self.cumulative_sum == 50:
            reward = 1  # Player wins
            self.done = True
            return (
                np.array([self.cumulative_sum], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Game continues; switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1
        reward = 0  # No reward for a regular move
        return (
            np.array([self.cumulative_sum], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        return (
            f"Current cumulative sum: {self.cumulative_sum}, "
            f"Current player: Player {self.current_player}"
        )

    def valid_moves(self):
        # Valid actions are those that do not cause the sum to exceed 50
        valid_actions = [
            action for action in range(10) if self.cumulative_sum + (action + 1) <= 50
        ]
        return valid_actions
