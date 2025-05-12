import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Starting number is 60
        self.starting_number = 60

        # Define action and observation space
        # The action space consists of integers from 0 to 59, corresponding to divisors 1 to 60
        self.action_space = spaces.Discrete(60)

        # The observation is the current number
        self.observation_space = spaces.Box(
            low=2, high=self.starting_number, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self._get_obs(), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        # Check if current number is prime at the start of the turn
        if self._is_prime(self.current_number):
            self.done = True
            reward = -1  # Player loses because the number is prime
            return self._get_obs(), reward, True, False, {}

        # Map action to divisor (1 to 60)
        divisor = action + 1

        # Check if action is valid
        if self._is_valid_move(divisor):
            # Perform the division
            self.current_number = self.current_number // divisor

            # Check if the new current number is prime
            if self._is_prime(self.current_number):
                self.done = True
                reward = 1  # Player wins
                return self._get_obs(), reward, True, False, {}
            else:
                # Switch player and continue the game
                self.current_player = 2 if self.current_player == 1 else 1
                reward = 0  # No reward
                return self._get_obs(), reward, False, False, {}
        else:
            # Invalid move, player loses
            self.done = True
            reward = -10  # Penalty for invalid move
            return self._get_obs(), reward, True, False, {}

    def render(self):
        return f"Current Number: {self.current_number}\nCurrent Player: Player {self.current_player}"

    def valid_moves(self):
        # Return a list of valid actions (indices) corresponding to proper divisors
        proper_divisors = self._get_proper_divisors(self.current_number)
        return [d - 1 for d in proper_divisors]  # Adjust for zero-based indexing

    def _get_obs(self):
        return np.array([self.current_number], dtype=np.int32)

    def _is_valid_move(self, divisor):
        if divisor <= 1 or divisor >= self.current_number:
            return False
        if self.current_number % divisor != 0:
            return False
        return True

    def _get_proper_divisors(self, n):
        # Return proper divisors of n excluding 1 and n
        divisors = [i for i in range(2, n) if n % i == 0]
        return divisors

    def _is_prime(self, n):
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
