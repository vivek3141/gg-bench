import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N_start=20, N_max=100):
        super(CustomEnv, self).__init__()

        self.N_start = N_start
        self.N_max = N_max
        # Define action space: possible divisors from 0 to N_max - 1
        self.action_space = spaces.Discrete(N_max)
        # Define observation space: current number N
        self.observation_space = spaces.Box(
            low=1, high=self.N_max, shape=(1,), dtype=np.int32
        )

        self.current_N = None
        self.done = False
        self.current_player = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_N = self.N_start
        self.done = False
        self.current_player = 1  # 1 for Player 1, -1 for Player 2
        return np.array([self.current_N], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # No further moves allowed if the game is over
            return np.array([self.current_N], dtype=np.int32), -10, True, False, {}

        # Get valid moves for the current state
        valid_actions = self.valid_moves(self.current_N)

        if action not in valid_actions:
            # Invalid action
            self.done = True
            reward = -10
            return (
                np.array([self.current_N], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Valid action: subtract the chosen proper divisor
            self.current_N -= action

            # Check if opponent has any valid moves
            opponent_valid_actions = self.valid_moves(self.current_N)
            if not opponent_valid_actions:
                # Opponent cannot move; current player wins
                self.done = True
                reward = 1
                return (
                    np.array([self.current_N], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )
            else:
                # Game continues; switch to the other player
                self.current_player *= -1
                reward = -10  # Penalty for each valid move
                return (
                    np.array([self.current_N], dtype=np.int32),
                    reward,
                    False,
                    False,
                    {},
                )

    def render(self):
        player = "Player 1" if self.current_player == 1 else "Player 2"
        return f"Current number: {self.current_N} | Current player: {player}"

    def valid_moves(self, N):
        # Return a list of proper divisors of N
        return [d for d in range(2, N) if N % d == 0]

    def is_prime(self, n):
        # Check if a number is prime
        if n <= 1:
            return False
        elif n <= 3:
            return True
        elif n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
