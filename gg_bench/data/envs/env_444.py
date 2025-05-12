import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, N_init=15, N_max=100):
        super(CustomEnv, self).__init__()

        self.N_init = N_init
        self.N_max = N_max

        # Action space: Actions correspond to proper divisors from 1 to N_max
        self.action_space = spaces.Discrete(
            N_max + 1
        )  # Actions are integers from 0 to N_max

        # Observation space: The current N value (from 1 to N_max)
        self.observation_space = spaces.Box(low=1, high=N_max, shape=(), dtype=np.int32)

        # Initialize the environment
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.N_init
        self.done = False
        self.current_player = 1  # Player 1 starts
        info = {}
        return self.N, info  # Return observation and info

    def step(self, action):
        if self.done:
            raise Exception("Game is over. Please reset the environment.")

        # Check if there are valid moves
        valid_moves = self.valid_moves()
        if not valid_moves:
            # Current player cannot make a move and loses
            reward = -1
            self.done = True
            info = {"reason": "No valid moves. Player loses."}
            return (
                self.N,
                reward,
                self.done,
                False,
                info,
            )  # Observation, reward, terminated, truncated, info

        # Check if the action is valid
        if action not in valid_moves:
            # Invalid move
            reward = -10
            self.done = True
            info = {"reason": "Invalid move"}
            return self.N, reward, self.done, False, info

        # Apply the action
        self.N -= action

        # Check for win condition
        if self.N == 1:
            reward = 1  # Current player wins
            self.done = True
            info = {"winner": self.current_player}
            return self.N, reward, self.done, False, info

        # No reward for a valid move that doesn't end the game
        reward = 0
        self.done = False
        info = {}
        return self.N, reward, self.done, False, info

    def render(self):
        return f"Current N is {self.N}"

    def valid_moves(self):
        # Return a list of valid actions (proper divisors of N)
        if self.N <= 1:
            return []
        # Proper divisors are positive integers less than N that divide N evenly
        proper_divisors = [d for d in range(1, self.N) if self.N % d == 0]
        return proper_divisors
