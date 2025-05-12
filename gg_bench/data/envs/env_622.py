import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Maximum value for N (shared number)
        self.max_n = 100  # You can adjust this as needed
        self.initial_n = 60  # Default initial N value

        # Define action and observation spaces
        # Action space: Discrete set of possible divisors from 2 to max_n (inclusive)
        self.action_space = spaces.Discrete(
            self.max_n - 1
        )  # Actions correspond to divisors 2 to max_n

        # Observation space: The current value of N
        self.observation_space = spaces.Box(
            low=1, high=self.max_n, shape=(1,), dtype=np.int64
        )

        # Initialize the shared number N and current player
        self.n = None  # Current value of N
        self.current_player = 1  # Player 1 starts

        self.done = False  # Game over flag

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "initial_n" in options:
            self.n = options["initial_n"]
        else:
            self.n = self.initial_n  # Use default initial N

        if self.n < 2 or self.n > self.max_n:
            raise ValueError(
                f"Initial N must be between 2 and {self.max_n}, but got {self.n}"
            )

        self.current_player = 1  # Reset to Player 1
        self.done = False

        return np.array([self.n], dtype=np.int64), {}  # Observation and info

    def step(self, action):

        if self.done:
            return np.array([self.n], dtype=np.int64), 0, True, False, {}

        # At the beginning of the player's turn, check if they have any valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Current player cannot move, they lose
            self.done = True
            reward = -10
            return np.array([self.n], dtype=np.int64), reward, True, False, {}

        # Check if the action is valid
        if action not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10
            return np.array([self.n], dtype=np.int64), reward, True, False, {}

        # Valid move
        divisor = action + 2  # Map action index to divisor
        self.n = self.n // divisor

        # Switch player
        self.current_player = (
            3 - self.current_player
        )  # Switch between Player 1 (1) and Player 2 (2)

        # After the move, check if the next player has valid moves
        next_valid_actions = self.valid_moves()
        if not next_valid_actions:
            # Next player cannot move, current player wins
            self.done = True
            reward = 1  # Current player wins
        else:
            # Game continues
            reward = 0

        return np.array([self.n], dtype=np.int64), reward, self.done, False, {}

    def render(self):
        return f"Current N: {self.n}, Current Player: {self.current_player}"

    def valid_moves(self):
        # Return a list of valid action indices
        proper_divisors = [d for d in range(2, self.n) if self.n % d == 0]
        action_indices = [
            d - 2 for d in proper_divisors
        ]  # Map divisors to action indices
        return action_indices
