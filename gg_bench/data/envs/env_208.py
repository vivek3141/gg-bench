import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Action space: 8 actions, corresponding to divisors from 2 to 9
        self.action_space = spaces.Discrete(8)

        # Observation space: The shared number N, between 1 and 100
        self.observation_space = spaces.Box(low=1, high=100, shape=(1,), dtype=np.int32)

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = 100
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Return observation and info

    def step(self, action):
        if self.done:
            return (
                np.array([self.N], dtype=np.int32),
                0,
                True,
                False,
                {"message": "Game already over"},
            )

        # Map action to divisor
        if action < 0 or action > 7:
            # Invalid action index
            self.done = True
            return (
                np.array([self.N], dtype=np.int32),
                -10,
                True,
                False,
                {"message": "Invalid action index"},
            )

        divisor = action + 2  # Actions 0-7 correspond to divisors 2-9

        # Check if divisor is valid
        if self.N < divisor or self.N % divisor != 0:
            # Invalid move
            self.done = True
            reward = -10
            info = {"message": "Invalid move"}
            return np.array([self.N], dtype=np.int32), reward, True, False, info

        # Valid move, perform division
        self.N = self.N // divisor

        if self.N == 1:
            # Current player wins
            self.done = True
            reward = 1
            info = {"message": f"Player {self.current_player} wins"}
            return np.array([self.N], dtype=np.int32), reward, True, False, info
        else:
            # Switch player
            self.current_player = 2 if self.current_player == 1 else 1
            reward = 0
            info = {}
            return np.array([self.N], dtype=np.int32), reward, False, False, info

    def render(self):
        return f"Current player: Player {self.current_player}, N: {self.N}"

    def valid_moves(self):
        # Return a list of valid moves as indices of the action_space
        valid_actions = []
        for action in range(8):  # Actions 0-7 correspond to divisors 2-9
            divisor = action + 2
            if self.N >= divisor and self.N % divisor == 0:
                valid_actions.append(action)
        return valid_actions
