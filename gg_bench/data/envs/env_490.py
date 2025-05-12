import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Default starting number (composite number greater than 1)
        self.max_shared_number = 60

        # Action space: possible integers from 0 to max_shared_number
        # Actions represent the chosen proper divisor
        # So action space is spaces.Discrete(self.max_shared_number + 1)
        self.action_space = spaces.Discrete(self.max_shared_number + 1)

        # Observation is the current shared number
        self.observation_space = spaces.Box(
            low=2, high=self.max_shared_number, shape=(1,), dtype=np.int32
        )

        self.shared_number = None
        self.current_player = None
        self.done = None

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "shared_number" in options:
            self.shared_number = options["shared_number"]
        else:
            self.shared_number = self.max_shared_number  # Default starting number

        if self.shared_number <= 1 or self.is_prime(self.shared_number):
            raise ValueError(
                "Starting shared number must be a composite number greater than 1."
            )

        self.current_player = 1  # Player 1 starts
        self.done = False

        return np.array([self.shared_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.shared_number], dtype=np.int32), 0, True, False, {}

        # Check if action is a valid proper divisor
        if (
            action <= 1
            or action >= self.shared_number
            or self.shared_number % action != 0
        ):
            self.done = True
            return (
                np.array([self.shared_number], dtype=np.int32),
                0,
                True,
                False,
                {"reason": "Invalid move"},
            )

        # Valid move, apply penalty of -10
        self.shared_number = self.shared_number // action

        # Check if opponent can make a move (i.e., check if shared_number has proper divisors)
        opponent_valid_moves = self.get_proper_divisors(self.shared_number)

        if not opponent_valid_moves:
            # Opponent cannot make a move; current player wins
            self.done = True
            return np.array([self.shared_number], dtype=np.int32), +1, True, False, {}
        else:
            # Switch player
            self.current_player *= -1
            return np.array([self.shared_number], dtype=np.int32), -10, False, False, {}

    def render(self):
        render_str = f"Current Shared Number: {self.shared_number}\n"
        render_str += f"Player {self.current_player}'s Turn\n"
        valid_moves = self.get_proper_divisors(self.shared_number)
        render_str += f"Available Divisors: {', '.join(map(str, valid_moves))}\n"
        return render_str

    def valid_moves(self):
        return self.get_proper_divisors(self.shared_number)

    def get_proper_divisors(self, number):
        return [i for i in range(2, number) if number % i == 0]

    def is_prime(self, n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
