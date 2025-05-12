import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N=10):
        super(CustomEnv, self).__init__()

        self.N = N  # Maximum number in the game
        # Action space: numbers from 2 to N, indices 0 to N-2
        self.action_space = spaces.Discrete(self.N - 1)
        # Observation space: array representing availability of numbers from 2 to N
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.N - 1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.numbers = np.ones(self.N - 1, dtype=np.float32)
        self.current_player = 1  # Player 1 starts
        return self.numbers.copy(), {}

    def step(self, action):
        if action < 0 or action >= self.N - 1:
            # Invalid action
            return self.numbers.copy(), -10, True, False, {}

        if self.numbers[action] == 0:
            # Number already removed, invalid move
            return self.numbers.copy(), -10, True, False, {}

        # Valid action
        selected_number = action + 2  # The actual number

        # Remove selected number
        self.numbers[action] = 0

        # Remove proper divisors > 1
        for n in range(2, selected_number):
            if selected_number % n == 0:
                # n is a proper divisor
                idx = n - 2
                if self.numbers[idx] == 1:
                    self.numbers[idx] = 0

        # Check if game is over
        if np.sum(self.numbers) == 0:
            # Current player wins
            reward = 1
            done = True
        else:
            reward = -10
            done = False
            # Switch player
            self.current_player = 3 - self.current_player  # switch between 1 and 2

        return self.numbers.copy(), reward, done, False, {}

    def render(self):
        available_numbers = [
            str(i + 2) for i in range(self.N - 1) if self.numbers[i] == 1
        ]
        return f"Current player: Player {self.current_player}\nAvailable numbers: {', '.join(available_numbers)}"

    def valid_moves(self):
        return [i for i in range(self.N - 1) if self.numbers[i] == 1]
