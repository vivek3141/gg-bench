import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the perfect squares up to 100
        self.perfect_squares = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
        # Action space: Discrete actions corresponding to indices of perfect squares
        self.action_space = spaces.Discrete(len(self.perfect_squares))  # Actions 0-9

        # Observation space: [current N, current player (-1 or 1)]
        self.observation_space = spaces.Box(
            low=np.array([0, -1], dtype=np.int32),
            high=np.array([100, 1], dtype=np.int32),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = 100  # Starting number N
        self.current_player = 1  # Player 1 starts (1 or -1)
        self.done = False
        observation = np.array([self.N, self.current_player], dtype=np.int32)
        return observation, {}  # Return observation and info dictionary

    def step(self, action):
        if self.done:
            # Game has ended; no further actions allowed
            observation = np.array([self.N, self.current_player], dtype=np.int32)
            return observation, -10, True, False, {}  # Invalid move after game over

        # Validate action index
        if action < 0 or action >= len(self.perfect_squares):
            # Invalid action index
            observation = np.array([self.N, self.current_player], dtype=np.int32)
            self.done = True
            return observation, -10, True, False, {}  # Invalid action

        move_value = self.perfect_squares[action]

        # Check if move is valid (perfect square ≤ current N)
        if move_value > self.N:
            # Invalid move; cannot subtract a perfect square greater than N
            observation = np.array([self.N, self.current_player], dtype=np.int32)
            self.done = True
            return observation, -10, True, False, {}  # Invalid move

        # Subtract the chosen perfect square from N
        self.N -= move_value

        # Check for win condition
        if self.N == 0:
            # Current player wins
            self.done = True
            observation = np.array([self.N, self.current_player], dtype=np.int32)
            return observation, 1, True, False, {}  # Player wins

        # Swap to the other player
        self.current_player *= -1  # Switch between 1 and -1

        # Return the updated observation
        observation = np.array([self.N, self.current_player], dtype=np.int32)
        return observation, 0, False, False, {}  # Continue game

    def render(self):
        # Return a string representation of the current game state
        player = "1" if self.current_player == 1 else "2"
        return f"Current N: {self.N}, Player {player}'s turn."

    def valid_moves(self):
        # Return a list of valid action indices based on the current N
        valid_actions = []
        for idx, square in enumerate(self.perfect_squares):
            if square <= self.N:
                valid_actions.append(idx)
        return valid_actions
