import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to selecting numbers from 2 to 50 (inclusive)
        self.action_space = spaces.Discrete(
            49
        )  # Numbers 2 to 50 mapped to actions 0 to 48

        # Observation space:
        # Index 0: Current player (1 or 2)
        # Index 1: Current number (1 to 50)
        # Indices 2-50: Status of numbers 2 to 50 (0 if available, 1 if taken)
        self.observation_space = spaces.Box(low=0, high=50, shape=(51,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the Shared Number Pool (numbers from 2 to 50)
        self.available_numbers = list(range(2, 51))
        self.current_number = 1  # Starting with 1
        self.current_player = 1  # Player 1 starts
        self.terminated = False

        # Initialize observation
        self.observation = np.zeros(51, dtype=np.int32)
        self.observation[0] = self.current_player
        self.observation[1] = self.current_number
        # Indices 2 to 50 correspond to numbers 2 to 50; all set to 0 (available)
        self.observation[2:] = 0

        return self.observation, {}  # Return observation and info

    def step(self, action):
        if self.terminated:
            return self.observation, 0, True, False, {}  # Game is over

        # Map action to number n (numbers 2 to 50)
        n = action + 2

        # Check if action is valid
        if n not in self.available_numbers or not (
            self.current_number % n == 0 or n % self.current_number == 0
        ):
            self.terminated = True
            return self.observation, -10, True, False, {}

        # Valid move: update the game state
        self.available_numbers.remove(n)
        self.observation[n] = 1  # Mark number as taken
        self.current_number = n
        self.observation[1] = self.current_number

        # Check if opponent has any valid moves
        opponent_valid_moves = self.get_valid_moves()
        if not opponent_valid_moves:
            self.terminated = True
            return self.observation, 1, True, False, {}

        # Switch to the next player
        self.current_player = 2 if self.current_player == 1 else 1
        self.observation[0] = self.current_player

        return self.observation, -10, False, False, {}

    def get_valid_moves(self):
        valid_moves = []
        for num in self.available_numbers:
            if self.current_number % num == 0 or num % self.current_number == 0:
                valid_moves.append(num)
        return valid_moves

    def valid_moves(self):
        moves = []
        for num in self.available_numbers:
            if self.current_number % num == 0 or num % self.current_number == 0:
                action = num - 2  # Map number back to action index
                moves.append(action)
        return moves

    def render(self):
        s = f"--- Player {self.current_player}'s Turn ---\n"
        s += f"Current Number: {self.current_number}\n"
        s += "Available Numbers: "
        s += ", ".join(map(str, self.available_numbers))
        return s
