import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # There are 19 numbers from 2 to 20
        self.action_space = spaces.Discrete(19)

        # Observation is a vector of 19 elements (1 for available, 0 for removed)
        self.observation_space = spaces.Box(low=0, high=1, shape=(19,), dtype=np.int8)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.number_pool = np.ones(19, dtype=np.int8)
        self.current_player = 1
        self.done = False
        return self.number_pool.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.number_pool.copy(), 0, True, False, {}

        # Check if action is valid
        if action < 0 or action >= 19 or self.number_pool[action] == 0:
            self.done = True
            return self.number_pool.copy(), -10, True, False, {}

        # Action is valid
        number_selected = action + 2  # Map action index to number

        # Process the action according to the game rules
        if self.is_prime(number_selected):
            # Remove the prime and its multiples
            for i in range(19):
                num = i + 2
                if num % number_selected == 0 and self.number_pool[i] == 1:
                    self.number_pool[i] = 0
        else:
            # Remove its prime factors
            prime_factors = self.get_prime_factors(number_selected)
            for prime in prime_factors:
                idx = prime - 2
                if 0 <= idx < 19 and self.number_pool[idx] == 1:
                    self.number_pool[idx] = 0

        # Check if game is over
        if np.sum(self.number_pool) == 0:
            reward = 1
            self.done = True
            return self.number_pool.copy(), reward, True, False, {}

        # Switch to the next player
        self.current_player *= -1
        return self.number_pool.copy(), 0, False, False, {}

    def render(self):
        available_numbers = [i + 2 for i in range(19) if self.number_pool[i] == 1]
        print(f"Current Number Pool: {available_numbers}")

    def valid_moves(self):
        return [i for i in range(19) if self.number_pool[i] == 1]

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

    def get_prime_factors(self, n):
        factors = []
        # Check for 2
        if n % 2 == 0:
            factors.append(2)
            while n % 2 == 0:
                n = n // 2
        # Check for odd primes
        i = 3
        while i * i <= n:
            if n % i == 0:
                if i not in factors:
                    factors.append(i)
                while n % i == 0:
                    n = n // i
            i += 2
        if n > 2:
            if n not in factors:
                factors.append(n)
        return factors
