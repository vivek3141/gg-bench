import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # The action_space corresponds to possible divisors D from 2 to 50 inclusive
        self.action_space = spaces.Discrete(
            49
        )  # Actions 0 to 48 correspond to D=2 to D=50

        # The observation_space is the current Pile value
        self.observation_space = spaces.Box(low=1, high=100, shape=(1,), dtype=np.int32)

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pile = 100
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.info = {}
        return np.array([self.pile], dtype=np.int32), self.info  # Observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, no further moves can be made
            return (
                np.array([self.pile], dtype=np.int32),
                0,
                True,
                False,
                self.info,
            )

        # Map action to divisor D
        D = action + 2  # Actions 0-48 correspond to D=2-50

        # Validate the move according to the game rules
        if D <= 1 or D >= self.pile or D > self.pile // 2 or self.pile % D != 0:
            # Invalid move
            reward = -10
            self.done = True
            return (
                np.array([self.pile], dtype=np.int32),
                reward,
                True,
                False,
                self.info,
            )
        else:
            # Valid move
            self.pile = self.pile // D

            # Check if the player wins by reducing the pile to exactly 1
            if self.pile == 1:
                reward = 1  # Current player wins
                self.done = True
                return (
                    np.array([self.pile], dtype=np.int32),
                    reward,
                    True,
                    False,
                    self.info,
                )

            # Check if opponent has any valid moves
            if not self._has_valid_moves():
                # Opponent cannot move; current player wins
                reward = 1
                self.done = True
                return (
                    np.array([self.pile], dtype=np.int32),
                    reward,
                    True,
                    False,
                    self.info,
                )

            # Game continues
            reward = 0
            self.current_player *= -1  # Switch to the other player
            return (
                np.array([self.pile], dtype=np.int32),
                reward,
                False,
                False,
                self.info,
            )

    def render(self):
        return f"Current Pile: {self.pile}, Player {1 if self.current_player == 1 else 2}'s turn."

    def valid_moves(self):
        # Returns a list of valid action indices for the current player
        valid_actions = []
        for D in range(2, min(self.pile // 2 + 1, self.pile)):
            if self.pile % D == 0:
                action = D - 2  # Map D back to action index
                valid_actions.append(action)
        return valid_actions

    def _has_valid_moves(self):
        # Helper function to check if the next player has valid moves
        for D in range(2, min(self.pile // 2 + 1, self.pile)):
            if self.pile % D == 0:
                return True
        return False
