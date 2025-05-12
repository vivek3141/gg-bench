import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.N_max = 30  # Maximum starting value of N
        # Define action space: actions correspond to numbers from 0 to N_max
        # Actions are interpreted as the divisor to subtract from N
        self.action_space = spaces.Discrete(self.N_max + 1)
        # Observation space is the current value of N
        self.observation_space = spaces.Box(
            low=2, high=self.N_max, shape=(1,), dtype=np.int32
        )

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "starting_N" in options:
            self.N = options["starting_N"]
            if self.N < 2 or self.N > self.N_max:
                raise ValueError(
                    f"starting_N must be between 2 and {self.N_max}, inclusive."
                )
        else:
            self.N = np.random.randint(10, self.N_max + 1)
        self.current_player = 1  # 1 or -1 to represent two players
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                np.array([self.N], dtype=np.int32),
                0,
                True,
                False,
                {},
            )  # Game is over

        valid_divisors = self.get_proper_divisors(self.N)

        if action not in valid_divisors:
            # Invalid move
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Valid move
        self.N -= action

        # Check if the next player can move
        next_valid_divisors = self.get_proper_divisors(self.N)
        if not next_valid_divisors:
            # Next player cannot move; current player wins
            self.done = True
            reward = 1  # Current player wins
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Game continues
        self.current_player *= -1  # Switch player
        reward = 0
        return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def render(self):
        valid_divisors = self.get_proper_divisors(self.N)
        render_str = f"Current N: {self.N}\n"
        render_str += f"Current player: Player {1 if self.current_player == 1 else 2}\n"
        render_str += (
            f"Proper divisors (valid moves, excluding 1 and N): {valid_divisors}\n"
        )
        print(render_str)

    def valid_moves(self):
        return self.get_proper_divisors(self.N)

    def get_proper_divisors(self, N):
        divisors = []
        for i in range(2, N):
            if N % i == 0:
                divisors.append(i)
        return divisors
