import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: indices corresponding to factors 2, 3, 5
        self.action_space = spaces.Discrete(3)

        # Define observation space: current_number (int) and current_player (0 or 1)
        self.observation_space = spaces.Box(
            low=np.array([1, 0]), high=np.array([1000, 1]), dtype=np.int64
        )

        # Map action indices to factors
        self.factors = [2, 3, 5]

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 0  # 0: Player 1, 1: Player 2
        self.done = False
        return (
            np.array([self.current_number, self.current_player], dtype=np.int64),
            {},
        )

    def step(self, action):
        if self.done:
            # If the game is already over, return the current state
            return (
                np.array([self.current_number, self.current_player], dtype=np.int64),
                0,
                True,
                False,
                {},
            )

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move: action not in valid moves
            self.done = True
            reward = -10
            return (
                np.array([self.current_number, self.current_player], dtype=np.int64),
                reward,
                True,
                False,
                {},
            )

        # Apply the chosen factor
        factor = self.factors[action]
        self.current_number *= factor

        # Check for win or loss conditions
        if self.current_number == 100:
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_number, self.current_player], dtype=np.int64),
                reward,
                True,
                False,
                {},
            )
        elif self.current_number > 100:
            # Current player loses
            self.done = True
            reward = -10
            return (
                np.array([self.current_number, self.current_player], dtype=np.int64),
                reward,
                True,
                False,
                {},
            )
        else:
            # Switch to the next player
            self.current_player = 1 - self.current_player

            # Check if the next player has any valid moves
            if not self.valid_moves():
                # Next player cannot make a valid move, current player wins
                self.done = True
                reward = 1
                return (
                    np.array(
                        [self.current_number, self.current_player], dtype=np.int64
                    ),
                    reward,
                    True,
                    False,
                    {},
                )
            else:
                # Continue the game
                reward = 0
                return (
                    np.array(
                        [self.current_number, self.current_player], dtype=np.int64
                    ),
                    reward,
                    False,
                    False,
                    {},
                )

    def render(self):
        # Return a string representation of the current state
        return f"Current number: {self.current_number}, Current player: Player {self.current_player + 1}"

    def valid_moves(self):
        # Return a list of valid action indices
        valid_actions = []
        for idx, factor in enumerate(self.factors):
            potential_number = self.current_number * factor
            if potential_number <= 100:
                valid_actions.append(idx)
        return valid_actions
