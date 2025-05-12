import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Game parameters
        self.TARGET_LENGTH = 10  # Chain target length
        self.MAX_NUMBER = 1000  # Maximum number allowed in the game

        # Define action and observation space
        self.action_space = spaces.Discrete(
            self.MAX_NUMBER + 1
        )  # Actions are numbers from 0 to MAX_NUMBER

        # Observation is the chain represented as a fixed-size array
        # Unused positions are filled with -1
        self.observation_space = spaces.Box(
            low=-1, high=self.MAX_NUMBER, shape=(self.TARGET_LENGTH,), dtype=np.int32
        )

        # Initialize game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.chain = -1 * np.ones(
            self.TARGET_LENGTH, dtype=np.int32
        )  # Initialize chain with -1
        self.chain[0] = 1  # Starting number is always 1
        self.chain_length = 1  # Current length of the chain
        self.used_numbers = set([1])  # Set of numbers already used in the chain
        self.current_player = 1  # Player 1 starts
        self.done = False
        return self.chain.copy(), {}  # Return observation and info

    def step(self, action):
        reward = 0
        truncated = False
        info = {}

        # Check if the game is already over
        if self.done:
            return self.chain.copy(), reward, True, truncated, info

        # Check if action is within valid range
        if action < 2 or action > self.MAX_NUMBER:
            reward = -10
            self.done = True
            return self.chain.copy(), reward, True, truncated, info

        # Check if action has been used before
        if action in self.used_numbers:
            reward = -10
            self.done = True
            return self.chain.copy(), reward, True, truncated, info

        previous_number = self.chain[self.chain_length - 1]

        # Check if action is greater than previous number
        if action <= previous_number:
            reward = -10
            self.done = True
            return self.chain.copy(), reward, True, truncated, info

        # Check validity criteria
        is_divisible = action % previous_number == 0
        shares_digit = self._shares_common_digit(action, previous_number)

        if not (is_divisible or shares_digit):
            reward = -10
            self.done = True
            return self.chain.copy(), reward, True, truncated, info

        # Valid move
        self.chain[self.chain_length] = action
        self.chain_length += 1
        self.used_numbers.add(action)

        # Check for victory
        if self.chain_length == self.TARGET_LENGTH:
            reward = 1
            self.done = True
            return self.chain.copy(), reward, True, truncated, info

        # Switch to next player
        self.current_player *= -1
        return self.chain.copy(), reward, False, truncated, info

    def render(self):
        # Return a string representation of the current chain
        chain_str = "Current Chain: " + ", ".join(
            [str(num) for num in self.chain if num != -1]
        )
        return chain_str

    def valid_moves(self):
        # Return a list of valid moves for the current player
        valid_moves = []
        previous_number = self.chain[self.chain_length - 1]
        for number in range(previous_number + 1, self.MAX_NUMBER + 1):
            if number not in self.used_numbers:
                is_divisible = number % previous_number == 0
                shares_digit = self._shares_common_digit(number, previous_number)
                if is_divisible or shares_digit:
                    valid_moves.append(number)
        return valid_moves

    def _shares_common_digit(self, a, b):
        return bool(set(str(a)) & set(str(b)))
