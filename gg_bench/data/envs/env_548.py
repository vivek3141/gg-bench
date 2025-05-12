import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions: 0 - Subtract 1, 1-10 - Divide by prime factors (up to 10)
        self.action_space = spaces.Discrete(11)
        # Observation: Current value of N
        self.observation_space = spaces.Box(low=1, high=1e6, shape=(1,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = 100  # Starting number
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.action_map = {}  # Mapping from action indices to actions
        return np.array([self.N], dtype=np.int32), {}  # Observation and info

    def step(self, action):
        # Ensure valid_moves and action_map are current
        valid_actions = self.valid_moves()

        if action not in valid_actions or self.done:
            # Invalid action or game already over
            reward = -10
            terminated = True
            truncated = False
            return np.array([self.N], dtype=np.int32), reward, terminated, truncated, {}
        else:
            action_type = self.action_map[action]
            if action_type == "subtract_1":
                self.N -= 1
            elif isinstance(action_type, tuple) and action_type[0] == "divide":
                p = action_type[1]
                self.N = self.N // p
            else:
                # Invalid action type (should not occur)
                reward = -10
                terminated = True
                truncated = False
                return (
                    np.array([self.N], dtype=np.int32),
                    reward,
                    terminated,
                    truncated,
                    {},
                )

            # Check for win condition
            if self.N == 1:
                reward = 1  # Current player wins
                terminated = True
                truncated = False
            else:
                reward = 0  # Continue game
                terminated = False
                truncated = False
                # Switch to the other player
                self.current_player *= -1

            return np.array([self.N], dtype=np.int32), reward, terminated, truncated, {}

    def render(self):
        # Visual representation of the game state
        return f"Current N: {self.N}, Player: {self.current_player}"

    def valid_moves(self):
        # Generate the list of valid action indices and the action mapping
        valid_actions = []
        self.action_map = {}  # Reset action mapping

        if self.N > 1:
            # Action 0: Subtract 1
            valid_actions.append(0)
            self.action_map[0] = "subtract_1"
            # Get unique prime factors of N
            prime_factors = self.get_prime_factors(self.N)
            prime_factors = sorted(list(set(prime_factors)))
            for idx, p in enumerate(prime_factors):
                if idx < 10:  # Actions 1 to 10 for prime factors
                    action_idx = idx + 1
                    valid_actions.append(action_idx)
                    self.action_map[action_idx] = ("divide", p)
        else:
            # N == 1; no valid moves
            pass

        return valid_actions

    @staticmethod
    def get_prime_factors(n):
        # Compute the list of prime factors of n
        factors = []
        i = 2
        while i * i <= n:
            if n % i:
                i += 1
            else:
                factors.append(i)
                n = n // i
        if n > 1:
            factors.append(n)
        return factors
