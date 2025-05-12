import numpy as np
import gymnasium as gym
from gymnasium import spaces


def is_prime(n):
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


def get_proper_divisors(n):
    divisors = set()
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            if i != n // i:
                divisors.add(n // i)
    divisors.discard(n)
    return sorted(divisors)


class CustomEnv(gym.Env):
    def __init__(self, starting_N=30):
        super(CustomEnv, self).__init__()

        self.starting_N = starting_N
        self.action_space = spaces.Discrete(self.starting_N + 1)
        self.observation_space = spaces.Box(
            low=1, high=self.starting_N, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.starting_N
        self.current_player = 1
        self.done = False
        observation = np.array([self.N], dtype=np.int32)
        return observation, {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.N], dtype=np.int32),
                0,
                True,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            self.done = True
            return (
                np.array([self.N], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        self.N -= action
        observation = np.array([self.N], dtype=np.int32)

        if self.N == 1:
            self.done = True
            return observation, 1, True, False, {}
        else:
            self.current_player *= -1
            return observation, 0, False, False, {}

    def render(self):
        state_str = f"Current Target Number: {self.N}\n"
        state_str += f"Player {self.current_player}'s Turn.\n"
        return state_str

    def valid_moves(self):
        if is_prime(self.N):
            return [1]
        else:
            divisors = get_proper_divisors(self.N)
            return divisors
