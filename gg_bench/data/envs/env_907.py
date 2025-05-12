import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.MAX_NUMBER = 1000  # Maximum possible current number
        self.MAX_ACTION = 999  # Actions correspond to divisors from 2 to 1000

        # Define action and observation space
        self.action_space = spaces.Discrete(
            self.MAX_ACTION
        )  # Action indices from 0 to 998
        self.observation_space = spaces.Box(
            low=2, high=self.MAX_NUMBER, shape=(), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 60  # Default starting number can be changed
        self.current_player = 1  # Player 1 starts; Player 1 = 1, Player 2 = -1
        self.done = False
        return self.current_number, {}  # Return observation and info

    def step(self, action):
        if self.done:
            return self.current_number, 0, True, False, {}

        divisor = action + 2  # Map action index back to divisor
        # Check for valid move
        if divisor <= 1 or divisor >= self.current_number:
            # Invalid move: not a proper divisor
            reward = -10
            self.done = True
            return self.current_number, reward, True, False, {}
        if self.current_number % divisor != 0:
            # Invalid move: divisor does not divide current_number
            reward = -10
            self.done = True
            return self.current_number, reward, True, False, {}

        # Valid move
        self.current_number = self.current_number // divisor

        # Check if the next player has any valid moves
        proper_divisors = self.get_proper_divisors(self.current_number)
        if not proper_divisors:
            # Next player cannot make a valid move; current player wins
            reward = 1  # Win reward
            self.done = True
            return self.current_number, reward, True, False, {}

        # Switch player
        self.current_player *= -1
        reward = 0  # No immediate reward
        return self.current_number, reward, False, False, {}

    def render(self):
        # Return a string representing the current state
        proper_divisors = self.get_proper_divisors(self.current_number)
        divisors_str = ", ".join(map(str, proper_divisors))
        return (
            f"Player {1 if self.current_player == 1 else 2}'s turn.\n"
            f"Current Number: {self.current_number}\n"
            f"Proper Divisors: {divisors_str}\n"
        )

    def valid_moves(self):
        # Return list of valid action indices
        proper_divisors = self.get_proper_divisors(self.current_number)
        action_indices = [
            d - 2 for d in proper_divisors
        ]  # Map divisors back to action indices
        return action_indices

    def get_proper_divisors(self, n):
        # Return list of proper divisors of n (excluding 1 and n)
        divisors = []
        for i in range(2, n):
            if n % i == 0:
                divisors.append(i)
        return divisors
