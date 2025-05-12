import numpy as np
import gymnasium as gym
from gymnasium import spaces


def is_prime(n):
    """Check if the given number n is prime."""
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


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.MAX_N = 100  # Maximum starting number N
        self.action_space = spaces.Discrete(
            self.MAX_N + 1
        )  # Actions from 0 to MAX_N inclusive
        self.observation_space = spaces.Box(
            low=np.array([1, -1]),
            high=np.array([self.MAX_N, 1]),
            shape=(2,),
            dtype=np.int64,
        )

        self.N = None
        self.current_player = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = np.random.randint(2, self.MAX_N + 1)  # Starting N between 2 and MAX_N
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N, self.current_player]), {}  # Observation and info

    def step(self, action):
        if self.done:
            # If the game is already over
            return np.array([self.N, self.current_player]), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move
            self.done = True
            return (
                np.array([self.N, self.current_player]),
                -10,
                True,
                False,
                {},
            )

        # Perform the action
        if action == 0:
            # Subtract 1 from N
            self.N -= 1
        else:
            # Divide N by the chosen proper divisor
            self.N = self.N // action

        if self.N == 1:
            # Current player wins
            self.done = True
            return np.array([self.N, self.current_player]), 1, True, False, {}

        # Switch to the next player
        self.current_player *= -1

        return np.array([self.N, self.current_player]), 0, False, False, {}

    def render(self):
        return (
            f"Current N: {self.N}, Player: {'1' if self.current_player == 1 else '2'}"
        )

    def valid_moves(self):
        """Return a list of valid action indices for the current state."""
        moves = []
        if self.N == 1:
            # No valid moves when N is 1
            return moves

        if is_prime(self.N):
            # When N is prime, only action 0 (subtract 1) is valid
            moves.append(0)
        else:
            # When N is composite, list all proper divisors as valid actions
            for d in range(2, self.N):
                if self.N % d == 0:
                    moves.append(d)
        return moves
