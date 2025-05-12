import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space

        # Action space: Discrete(10), actions correspond to perfect squares from 1^2 to 10^2
        self.action_space = spaces.Discrete(10)

        # Observation space: Box(0, 100, (1,), int32), the current total
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)

        # Initialize state variables
        self.total = None
        self.current_player = None
        self.done = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total = 100  # Starting total
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.total], dtype=np.int32), {}  # observation, info

    def step(self, action):
        if self.done:
            # Game is already over
            return np.array([self.total], dtype=np.int32), -10, True, False, {}

        # Map action index to perfect square (action 0-9 correspond to squares 1^2 to 10^2)
        perfect_square = (action + 1) ** 2

        # Check if the action is valid
        if perfect_square > self.total:
            # Invalid move
            self.done = True
            return np.array([self.total], dtype=np.int32), -10, True, False, {}

        # Subtract the perfect square from the total
        self.total -= perfect_square

        if self.total == 0:
            # Current player wins
            self.done = True
            return np.array([self.total], dtype=np.int32), 1, True, False, {}

        # Check if the next player has any valid moves
        if not self.valid_moves():
            # Opponent cannot move; current player wins
            self.done = True
            return np.array([self.total], dtype=np.int32), 1, True, False, {}

        # Switch to the next player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2

        return np.array([self.total], dtype=np.int32), -10, False, False, {}

    def render(self):
        # Return a string representation of the current state
        return f"Current total: {self.total}, Current player: {self.current_player}"

    def valid_moves(self):
        # Return a list of valid action indices based on the current total
        valid_actions = []
        for action in range(10):
            perfect_square = (action + 1) ** 2
            if perfect_square <= self.total:
                valid_actions.append(action)
        return valid_actions
