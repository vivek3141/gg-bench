import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 - Double, 1 - Add One
        self.action_space = spaces.Discrete(2)

        # Observation space: cumulative number from 1 to 31
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([31]), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cumulative_number = 1
        self.current_player = 1  # Player 1: 1, Player 2: -1
        self.done = False
        return np.array([self.cumulative_number]), {}

    def step(self, action):
        if self.done:
            return np.array([self.cumulative_number]), 0, True, False, {}

        # Check if the action is valid
        if action not in [0, 1]:
            # Invalid action
            self.done = True
            return (
                np.array([self.cumulative_number]),
                -10,
                True,
                False,
                {"invalid_action": True},
            )

        # Determine the new cumulative number
        if action == 0:
            # Double
            new_cumulative_number = self.cumulative_number * 2
        elif action == 1:
            # Add One
            new_cumulative_number = self.cumulative_number + 1

        # Check if the action is valid (does not exceed target number)
        if new_cumulative_number > 31:
            # Invalid action
            self.done = True
            return (
                np.array([self.cumulative_number]),
                -10,
                True,
                False,
                {"invalid_action": True},
            )

        # Update cumulative number
        self.cumulative_number = new_cumulative_number

        # Check if the current player wins by reaching the target number
        if self.cumulative_number == 31:
            self.done = True
            return np.array([self.cumulative_number]), 1, True, False, {}

        # Check if the next player has any valid moves
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            # Next player has no valid moves; current player wins
            self.done = True
            return np.array([self.cumulative_number]), 1, True, False, {}

        # Switch current player
        self.current_player *= -1

        # Continue the game
        return np.array([self.cumulative_number]), 0, False, False, {}

    def render(self):
        return f"Current cumulative number: {self.cumulative_number}, Current player: {'Player 1' if self.current_player == 1 else 'Player 2'}"

    def valid_moves(self):
        moves = []
        if self.cumulative_number * 2 <= 31:
            moves.append(0)  # Double
        if self.cumulative_number + 1 <= 31:
            moves.append(1)  # Add One
        return moves
