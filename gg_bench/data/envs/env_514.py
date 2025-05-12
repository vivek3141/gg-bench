import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the maximum possible number N can be
        self.N_max = 1000  # Adjust as needed for your problem space
        self.initial_N = 60  # Default starting number N

        # Define action space: possible divisors from 2 up to N_max
        # Actions correspond to possible divisors D
        self.action_space = spaces.Discrete(
            self.N_max + 1
        )  # Actions are integers from 0 to N_max

        # Define observation space: the current value of N
        self.observation_space = spaces.Box(
            low=2, high=self.N_max, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.current_N = None
        self.current_player = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the game state
        self.current_N = self.initial_N
        self.current_player = 1  # Player 1 starts first
        self.done = False
        return (
            np.array([self.current_N], dtype=np.int32),
            {},
        )  # Return observation and info

    def step(self, action):
        D = action  # The action represents the chosen divisor D

        # Check if the game is already over
        if self.done:
            return (
                np.array([self.current_N], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Validate the action
        if D <= 1 or D >= self.current_N or self.current_N % D != 0:
            # Invalid move
            self.done = True
            return (
                np.array([self.current_N], dtype=np.int32),
                -10,  # Penalty for invalid move
                True,
                False,
                {},
            )

        # Valid move: update the current number N by dividing it by D
        self.current_N = self.current_N // D

        # Check if the game ends (opponent has no valid moves)
        if self.current_N <= 1:
            # Opponent cannot make a move, current player wins
            self.done = True
            return np.array([self.current_N], dtype=np.int32), 1, True, False, {}

        # Check if the opponent has valid moves
        opponent_valid_moves = self.valid_moves()
        if len(opponent_valid_moves) == 0:
            # Opponent cannot make a move, current player wins
            self.done = True
            return np.array([self.current_N], dtype=np.int32), 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        # Continue the game
        return (
            np.array([self.current_N], dtype=np.int32),  # Observation
            0,  # No reward for a regular valid move
            False,  # Game is not over
            False,
            {},  # Info dictionary
        )

    def render(self):
        # Return a string representation of the game state
        return f"Current N: {self.current_N}, Player: {self.current_player}"

    def valid_moves(self):
        # Return a list of valid divisors (actions) for the current N
        valid_divisors = []
        for D in range(2, self.current_N):
            if self.current_N % D == 0:
                valid_divisors.append(D)
        return valid_divisors
