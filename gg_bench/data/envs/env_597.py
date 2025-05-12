import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, initial_N=100):
        super(CustomEnv, self).__init__()

        self.MAX_N = 9999999999  # Maximum value for N
        self.MAX_DIGITS = len(str(self.MAX_N))  # Maximum number of digits in N
        self.initial_N = initial_N  # Initial value of N

        # Define action and observation spaces
        self.action_space = spaces.Discrete(10)  # Digits 0 through 9
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(self.MAX_DIGITS + 1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.N = self.initial_N
        self.done = False
        self.current_player = 1  # Player 1 starts the game

        observation = self.get_observation()
        return observation, {}  # Return observation and info

    def get_observation(self):
        # Extract the digits of N
        N_digits = [int(digit) for digit in str(self.N)]
        # Pad the digits array to match MAX_DIGITS
        padded_digits = [0] * (self.MAX_DIGITS - len(N_digits)) + N_digits
        # Create the observation array with N and its digits
        observation = np.array([self.N] + padded_digits, dtype=np.int32)
        return observation

    def step(self, action):
        if self.done:
            # If the game is already over
            return self.get_observation(), 0, True, False, {}

        largest_digit = max([int(digit) for digit in str(self.N)])

        # Check for invalid moves
        if action != largest_digit or action > self.N:
            self.done = True
            reward = -10  # Penalty for invalid move
            return self.get_observation(), reward, True, False, {}

        # Perform the subtraction
        self.N -= action

        # Check for win condition
        if self.N == 0:
            self.done = True
            reward = 1  # Reward for winning the game
            return self.get_observation(), reward, True, False, {}
        else:
            # Continue the game
            reward = 0  # No reward for a valid move that doesn't win the game
            self.current_player *= -1  # Switch players
            return self.get_observation(), reward, False, False, {}

    def render(self):
        # Provide a string representation of the current state
        return f"Current N: {self.N}\nCurrent Player: {self.current_player}"

    def valid_moves(self):
        # The only valid move is subtracting the largest digit
        largest_digit = max([int(digit) for digit in str(self.N)])
        return [largest_digit]
