import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to selecting a divisor D, where D = action + 2
        # D ranges from 2 to 100 (since N can be up to 100)
        self.max_D = 100
        self.action_space = spaces.Discrete(
            self.max_D - 1
        )  # Actions: 0 to 98 corresponding to D = 2 to 100

        # Observation space consists of [N, last_D_used]
        # N ranges from 1 to 100
        # last_D_used ranges from 0 to 100 (0 indicates no previous D)
        self.observation_space = spaces.Box(
            low=np.array([1, 0]), high=np.array([100, 100]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = 100  # Starting number N
        self.last_D_used = 0  # No last D used at the start
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.N, self.last_D_used], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            # If the game is already over, return the current state
            return (
                np.array([self.N, self.last_D_used], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        D = action + 2  # Map action to divisor D

        # Check if D is a valid move
        if D <= 1 or D >= self.N or self.N % D != 0 or D == self.last_D_used:
            self.done = True
            reward = -10  # Invalid move
            return (
                np.array([self.N, self.last_D_used], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Perform the division
        self.N = self.N // D
        self.last_D_used = D

        # Check for win condition (current player wins by reducing N to 1)
        if self.N == 1:
            self.done = True
            reward = 1  # Current player wins
            return (
                np.array([self.N, self.last_D_used], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check if the next player has any valid moves
        opponent_valid_moves = self._get_valid_moves()
        if not opponent_valid_moves:
            # Opponent cannot make a move; current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.N, self.last_D_used], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Game continues; switch player
        self.current_player *= -1
        reward = 0  # No reward for ongoing game
        return (
            np.array([self.N, self.last_D_used], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        return (
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
            f"Current Number (N): {self.N}\n"
            f"Last Divisor Used: {self.last_D_used}\n"
            f"Valid Divisors: {self._get_valid_divisors()}\n"
        )

    def valid_moves(self):
        # Returns a list of valid action indices for the current player
        valid_moves = [
            D - 2 for D in range(2, self.N) if self.N % D == 0 and D != self.last_D_used
        ]
        return valid_moves

    def _get_valid_moves(self):
        # Helper function to get valid moves for the next player without switching state
        valid_moves = [
            D - 2 for D in range(2, self.N) if self.N % D == 0 and D != self.last_D_used
        ]
        return valid_moves

    def _get_valid_divisors(self):
        # Helper function to get valid divisors for rendering
        valid_divisors = [
            D for D in range(2, self.N) if self.N % D == 0 and D != self.last_D_used
        ]
        return valid_divisors
