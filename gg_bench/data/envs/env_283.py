import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to multipliers 2, 3, 4
        self.action_space = spaces.Discrete(
            3
        )  # Actions: 0, 1, 2 mapping to multipliers [2, 3, 4]

        # Observation includes cumulative number and current player (1 or 2)
        self.observation_space = spaces.Box(
            low=np.array([1, 1]), high=np.array([100, 2]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_number = 1  # Starting cumulative number
        self.done = False  # Game is not over
        self.current_player = 1  # Player 1 starts
        return (
            np.array([self.cumulative_number, self.current_player]),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            # The game is already over
            return (
                np.array([self.cumulative_number, self.current_player]),
                0,
                True,
                False,
                {},
            )

        # Map action index to multiplier
        multipliers = [2, 3, 4]
        multiplier = multipliers[action]

        # Update cumulative number
        new_cumulative_number = self.cumulative_number * multiplier

        # Check for win or loss
        if new_cumulative_number == 100:
            # Current player wins
            self.cumulative_number = new_cumulative_number
            self.done = True
            reward = 1
        elif new_cumulative_number > 100:
            # Current player loses
            self.cumulative_number = new_cumulative_number
            self.done = True
            reward = -10
        else:
            # Game continues
            self.cumulative_number = new_cumulative_number
            # Switch player
            self.current_player = 3 - self.current_player  # Switches between 1 and 2
            reward = 0

        return (
            np.array([self.cumulative_number, self.current_player]),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        state_str = f"Current Number: {self.cumulative_number}\n"
        state_str += f"Current Player: Player {self.current_player}"
        return state_str

    def valid_moves(self):
        if self.done:
            return []
        else:
            return [0, 1, 2]  # Actions corresponding to multipliers [2, 3, 4]
