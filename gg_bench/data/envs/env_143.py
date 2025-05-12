import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, initial_N=16):
        super(CustomEnv, self).__init__()

        self.initial_N = initial_N
        self.N = initial_N

        # Define the action space: possible divisors from 0 to initial_N
        self.action_space = spaces.Discrete(
            self.initial_N + 1
        )  # Actions from 0 to initial_N

        # Define the observation space: current value of N
        self.observation_space = spaces.Box(
            low=1, high=self.initial_N, shape=(1,), dtype=np.int32
        )

        # Initialize game state variables
        self.current_player = 1  # Player 1 or -1
        self.done = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.N = self.initial_N
        self.current_player = 1
        self.done = False
        return np.array([self.N], dtype=np.int32), {}  # Observation and info

    def get_proper_divisors(self, n):
        # Helper function to get proper divisors excluding 1 and n
        divisors = []
        for i in range(2, n):  # Exclude 1 and n itself
            if n % i == 0:
                divisors.append(i)
        return divisors

    def valid_moves(self):
        # Return a list of valid moves (proper divisors of N)
        return self.get_proper_divisors(self.N)

    def step(self, action):
        if self.done:
            # If the game has already ended
            return (
                np.array([self.N], dtype=np.int32),
                0,
                True,
                False,
                {},
            )  # No reward for actions after game end

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move: current player loses
            self.done = True
            reward = -10
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        # Valid move: subtract the chosen divisor from N
        self.N -= action

        # Check if the opponent can make a move
        next_valid_moves = self.get_proper_divisors(self.N)

        if self.N == 1 or len(next_valid_moves) == 0:
            # Opponent cannot move: current player wins
            self.done = True
            reward = 1
            return np.array([self.N], dtype=np.int32), reward, True, False, {}

        else:
            # Game continues: switch to the next player
            self.current_player *= -1  # Switch player internally
            reward = -10  # Penalty for making a valid move
            return np.array([self.N], dtype=np.int32), reward, False, False, {}

    def render(self):
        # Return a string representation of the game state
        state_str = f"Current N: {self.N}\n"
        state_str += f"Valid moves: {self.valid_moves()}\n"
        state_str += f"Player {1 if self.current_player == 1 else 2}'s turn"
        return state_str
