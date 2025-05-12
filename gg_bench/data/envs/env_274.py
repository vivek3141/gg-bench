import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define maximum divisor to consider for the action space
        self.MAX_D = 1000  # Maximum possible divisor
        # Actions correspond to divisors from 2 up to MAX_D
        # Since actions are indexed from 0, action 0 corresponds to divisor 2
        self.action_space = spaces.Discrete(self.MAX_D - 1)  # Actions: 0 to MAX_D - 2

        # Observation space is the current value of N
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([np.iinfo(np.int32).max]), dtype=np.int32
        )

        self.N_initial = 100  # Starting number N

        # Initialize environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "N" in options:
            self.N_initial = options["N"]
        self.N = self.N_initial
        self.done = False
        self.current_player = 1  # Player 1 starts
        return np.array([self.N], dtype=np.int32), {}  # Observation, info

    def step(self, action):
        if self.done:
            return np.array([self.N], dtype=np.int32), 0, True, False, {}

        d = (
            action + 2
        )  # Map action index to divisor (since action 0 corresponds to divisor 2)

        # Check if action (divisor d) is valid
        if d <= 1 or d > int(np.sqrt(self.N)) or self.N % d != 0:
            # Invalid move
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int32), reward, True, False, {}
        else:
            # Valid move
            self.N = self.N // d
            # Check if opponent has valid moves
            if not self.has_valid_moves():
                # Current player wins
                self.done = True
                reward = 1
                return np.array([self.N], dtype=np.int32), reward, True, False, {}
            else:
                # Switch to next player
                self.current_player = 3 - self.current_player  # Switch between 1 and 2
                reward = 0
                return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def has_valid_moves(self):
        max_d = int(np.sqrt(self.N))
        for d in range(2, max_d + 1):
            if self.N % d == 0:
                return True
        return False

    def valid_moves(self):
        # Returns list of valid action indices
        valid_actions = []
        max_d = int(np.sqrt(self.N))
        for d in range(2, max_d + 1):
            if self.N % d == 0:
                action = d - 2  # Map divisor d to action index
                valid_actions.append(action)
        return valid_actions

    def render(self):
        valid_actions = self.valid_moves()
        valid_divisors = [action + 2 for action in valid_actions]
        return f"Current N: {self.N}, Player {self.current_player}'s turn, Valid divisors: {valid_divisors}"
