import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, target_number=100):
        super(CustomEnv, self).__init__()

        self.target_number = target_number

        # Define action and observation space
        # Actions: 0 -> multiply by 2, 1 -> multiply by 3
        self.action_space = spaces.Discrete(2)

        # Observation: [current_number, current_player]
        # current_number: integer >= 1
        # current_player: -1 or 1
        self.observation_space = spaces.Box(
            low=np.array([1, -1]), high=np.array([np.inf, 1]), dtype=np.float64
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts (1 or -1)
        self.done = False

        return (
            np.array([self.current_number, self.current_player], dtype=np.float64),
            {},
        )

    def step(self, action):
        if action not in [0, 1] or self.done:
            # Invalid action or game already over
            return (
                np.array([self.current_number, self.current_player], dtype=np.float64),
                -10,
                True,
                False,
                {},
            )

        multiplier = 2 if action == 0 else 3

        # Update the current number
        self.current_number *= multiplier

        # Check for win condition
        if self.current_number >= self.target_number:
            self.done = True
            # Current player wins
            return (
                np.array([self.current_number, self.current_player], dtype=np.float64),
                1,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player *= -1

        return (
            np.array([self.current_number, self.current_player], dtype=np.float64),
            0,
            False,
            False,
            {},
        )

    def render(self):
        return f"Current number: {self.current_number}, Current player: {self.current_player}"

    def valid_moves(self):
        return [0, 1]
