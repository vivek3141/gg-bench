import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action space: 0 (Add 1), 1 (Multiply by 2)
        self.action_space = spaces.Discrete(2)

        # Define observation space: Current number between 1 and 20
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([20]), shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}  # Observation, info

    def step(self, action):
        if self.done:
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Check for available valid moves at the start of the turn
        valid_moves = self.valid_moves()
        if len(valid_moves) == 0:
            # Current player has no valid moves and loses the game
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -1,
                True,
                False,
                {},
            )

        # Validate the action
        if action not in valid_moves:
            # Invalid move results in an immediate loss
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Apply the chosen valid action
        if action == 0:
            # Add 1
            self.current_number += 1
        elif action == 1:
            # Multiply by 2
            self.current_number *= 2

        # Check for win condition
        if self.current_number == 20:
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

        return (
            np.array([self.current_number], dtype=np.int32),
            0,
            False,
            False,
            {},
        )

    def render(self):
        return f"Current Number: {self.current_number}\n"

    def valid_moves(self):
        moves = []
        if self.current_number + 1 <= 20:
            moves.append(0)  # Add 1
        if self.current_number * 2 <= 20:
            moves.append(1)  # Multiply by 2
        return moves
