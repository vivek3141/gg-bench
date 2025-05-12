import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define multipliers corresponding to actions
        self.multipliers = [2, 3, 4, 5, 6, 7, 8, 9]  # Actions map to these multipliers

        # Define action and observation space
        self.action_space = spaces.Discrete(
            len(self.multipliers)
        )  # Actions 0 to 7 correspond to multipliers 2 to 9

        # Observation space includes cumulative_value and current_player
        # cumulative_value: between 1 and 1000
        # current_player: -1 or 1
        self.observation_space = spaces.Box(
            low=np.array([1, -1]), high=np.array([1000, 1]), dtype=np.float32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_value = 1.0
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.cumulative_value, self.current_player], dtype=np.float32),
            {},
        )

    def valid_moves(self):
        valid_actions = []
        for i, multiplier in enumerate(self.multipliers):
            potential_value = self.cumulative_value * multiplier
            if potential_value <= 1000:
                valid_actions.append(i)
        return valid_actions

    def step(self, action):
        if self.done:
            return (
                np.array(
                    [self.cumulative_value, self.current_player], dtype=np.float32
                ),
                0,
                True,
                False,
                {},
            )

        # Check if the current player has any valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Current player loses because no valid moves
            self.done = True
            reward = 0
            return (
                np.array(
                    [self.cumulative_value, self.current_player], dtype=np.float32
                ),
                reward,
                True,
                False,
                {},
            )

        # Validate the action
        if action not in valid_actions:
            # Invalid move; game ends
            self.done = True
            reward = -10
            return (
                np.array(
                    [self.cumulative_value, self.current_player], dtype=np.float32
                ),
                reward,
                True,
                False,
                {},
            )

        # Apply the action
        multiplier = self.multipliers[action]
        self.cumulative_value *= multiplier

        # Check for winning condition
        if self.cumulative_value == 1000:
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array(
                    [self.cumulative_value, self.current_player], dtype=np.float32
                ),
                reward,
                True,
                False,
                {},
            )

        # Check for losing condition (exceeded 1000)
        if self.cumulative_value > 1000:
            # Current player loses
            self.done = True
            reward = 0
            return (
                np.array(
                    [self.cumulative_value, self.current_player], dtype=np.float32
                ),
                reward,
                True,
                False,
                {},
            )

        # Valid move; game continues
        reward = -10  # Penalize for valid move as per game rules

        # Switch current player
        self.current_player *= -1

        return (
            np.array([self.cumulative_value, self.current_player], dtype=np.float32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        player_str = "Player 1" if self.current_player == 1 else "Player 2"
        return (
            f"Current player: {player_str}, Cumulative Value: {self.cumulative_value}"
        )

    def valid_moves(self):
        valid_actions = []
        for i, multiplier in enumerate(self.multipliers):
            potential_value = self.cumulative_value * multiplier
            if potential_value <= 1000:
                valid_actions.append(i)
        return valid_actions
