import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions 0-4 correspond to adding numbers 1-5
        self.action_space = spaces.Discrete(5)

        # Observation space: shared total and current player (-1 or 1)
        self.observation_space = spaces.Box(
            low=np.array([0, -1]), high=np.array([1000, 1]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total = 0  # Shared total starts at 0
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game is not over
        return (
            np.array([self.total, self.current_player], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        # Check if the game has already ended
        if self.done:
            return (
                np.array([self.total, self.current_player], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Validate action
        if action not in [0, 1, 2, 3, 4]:
            self.done = True
            reward = -10  # Invalid move penalty
            return (
                np.array([self.total, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Map action to value to add (actions 0-4 correspond to adding 1-5)
        value_added = action + 1
        self.total += value_added

        # Check for bomb explosion
        if self.total % 13 == 0:
            self.done = True
            reward = -1  # Current player loses
            return (
                np.array([self.total, self.current_player], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player *= -1

        # Continue the game
        reward = 0  # No reward for a regular valid move
        return (
            np.array([self.total, self.current_player], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        player = "Player 1" if self.current_player == 1 else "Player 2"
        return f"Shared Total: {self.total}, {player}'s turn."

    def valid_moves(self):
        # Return valid moves if the game is not over
        if self.done:
            return []
        else:
            return [0, 1, 2, 3, 4]  # Actions corresponding to numbers 1-5
