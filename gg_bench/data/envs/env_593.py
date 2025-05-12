import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Allowed prime numbers for division
        self.allowed_primes = [2, 3, 5, 7]

        # Define action space: actions correspond to indices of allowed_primes
        self.action_space = spaces.Discrete(len(self.allowed_primes))

        # Define observation space: the current value of N
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([np.iinfo(np.int32).max]), dtype=np.int32
        )

        # Starting number N
        self.initial_N = 60

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.initial_N  # Reset N to the starting number
        self.current_player = 1  # Player 1 starts
        self.done = False  # Game is not over
        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def step(self, action):
        if self.done:
            # If game is already over, return penalty
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        # Get the chosen prime based on the action index
        chosen_prime = self.allowed_primes[action]

        if self.N % chosen_prime != 0:
            # Invalid move: chosen prime does not divide N exactly
            self.done = True
            return np.array([self.N], dtype=np.int32), -10, True, False, {}

        # Valid move: divide N by the chosen prime
        self.N = self.N // chosen_prime

        if self.N == 1:
            # Current player wins by reducing N to 1
            self.done = True
            return np.array([self.N], dtype=np.int32), 1, True, False, {}
        else:
            # Check if the opponent can make a valid move
            opponent_can_move = any(self.N % p == 0 for p in self.allowed_primes)
            if not opponent_can_move:
                # Current player wins because opponent cannot make a valid move
                self.done = True
                return np.array([self.N], dtype=np.int32), 1, True, False, {}
            else:
                # Game continues: switch to the next player
                self.current_player = 3 - self.current_player  # Switch between 1 and 2
                return np.array([self.N], dtype=np.int32), 0, False, False, {}

    def render(self):
        # Return a string representation of the game state
        return f"Current N: {self.N}, Current Player: {self.current_player}"

    def valid_moves(self):
        # Return a list of valid action indices based on the current N
        return [i for i, p in enumerate(self.allowed_primes) if self.N % p == 0]
