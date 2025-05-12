import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: 0 (Add 1), 1 (Multiply by 2)
        self.action_space = spaces.Discrete(2)

        # Define observation space: Current number between 1 and 100 inclusive
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([100]), shape=(1,), dtype=np.int32
        )

        # Initialize the current number and other variables
        self.current_number = None
        self.current_player = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            # If called after the game is over, return current state
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid or losing move
            self.done = True
            reward = -10
            terminated = True
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                terminated,
                False,
                {},
            )

        # Apply the action
        if action == 0:
            # Add 1
            self.current_number += 1
        elif action == 1:
            # Multiply by 2
            self.current_number *= 2

        # Check for win condition
        if self.current_number == 100:
            # Current player wins
            self.done = True
            reward = 1
            terminated = True
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                terminated,
                False,
                {},
            )

        # Check if no valid moves left for the next player
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            # Next player has no valid moves and loses
            self.done = True
            reward = 1  # Current player wins
            terminated = True
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                terminated,
                False,
                {},
            )

        # Switch to next player's turn
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        reward = 0
        terminated = False
        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            terminated,
            False,
            {},
        )

    def render(self):
        return f"Current Number: {self.current_number}"

    def valid_moves(self):
        valid_moves = []
        if self.current_number + 1 <= 100:
            valid_moves.append(0)  # Action 0 corresponds to +1
        if self.current_number * 2 <= 100:
            valid_moves.append(1)  # Action 1 corresponds to *2
        return valid_moves
