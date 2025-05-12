import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sympy import primerange


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the list of prime numbers up to 100
        self.primes = list(primerange(2, 100))

        # Map actions to prime numbers
        self.action_space = spaces.Discrete(len(self.primes))

        # Observation space is the current number
        self.observation_space = spaces.Box(
            low=1, high=10000, shape=(1,), dtype=np.int32
        )

        self.current_number = None
        self.current_player = 1  # 1 or -1 to represent players
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start with a random number between 50 and 100
        self.current_number = np.array(
            [self.np_random.integers(50, 101)], dtype=np.int32
        )
        self.current_player = 1
        self.done = False
        return self.current_number.copy(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.current_number.copy(), 0, True, False, {}

        # Check if there are valid moves
        valid_actions = self.valid_moves()
        if not valid_actions:
            # Current player cannot make a move and loses
            self.done = True
            reward = -1  # Current player loses
            return self.current_number.copy(), reward, True, False, {}

        selected_prime = self.primes[action]

        if selected_prime not in valid_actions:
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return self.current_number.copy(), reward, True, False, {}

        # Valid move: divide the current number by the selected prime
        self.current_number[0] //= selected_prime

        if self.current_number[0] == 1:
            # Current player wins
            self.done = True
            reward = 1
            return self.current_number.copy(), reward, True, False, {}
        else:
            # Switch player
            self.current_player *= -1
            reward = 0
            return self.current_number.copy(), reward, False, False, {}

    def render(self):
        available_moves = self.valid_moves()
        render_str = (
            f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
            f"Current Number: {self.current_number[0]}\n"
            f"Available Prime Factors: {available_moves}\n"
        )
        return render_str

    def valid_moves(self):
        # Return a list of valid prime factors as action indices
        valid_actions = []
        for idx, prime in enumerate(self.primes):
            if self.current_number[0] % prime == 0:
                valid_actions.append(prime)
        return valid_actions
