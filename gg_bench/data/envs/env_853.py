import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        self.N_min = 3  # Minimum starting N (greater than 1)
        self.N_max = 1000  # Maximum starting N

        # Define action and observation spaces
        # Action space: integers from 0 to N_max - 1
        self.action_space = spaces.Discrete(self.N_max)
        self.observation_space = spaces.Box(
            low=0, high=self.N_max, shape=(1,), dtype=np.int32
        )

        # Initialize the environment state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "starting_N" in options:
            self.N = options["starting_N"]
        else:
            self.N = np.random.randint(self.N_min, self.N_max)
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.N], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            raise Exception("Game is over. Please reset the environment.")

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            reward = -10  # Penalize for invalid move
            self.done = True
            return (
                np.array([self.N], dtype=np.int32),
                reward,
                self.done,
                False,
                {},
            )  # Observation, reward, terminated, truncated, info

        # Subtract the selected proper divisor from N
        self.N -= action

        # Check for victory condition
        if self.N == 0:
            reward = 1  # Current player wins by reducing N to zero
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, self.done, False, {}

        # Check if the next player has any valid moves
        opponent_valid_moves = self.get_proper_divisors(self.N)
        if not opponent_valid_moves:
            reward = 1  # Current player wins as opponent cannot move
            self.done = True
            return np.array([self.N], dtype=np.int32), reward, self.done, False, {}

        # Switch to the next player
        self.current_player = 1 if self.current_player == 2 else 2

        return np.array([self.N], dtype=np.int32), 0, self.done, False, {}

    def render(self):
        state = f"Current N: {self.N}, Current Player: Player {self.current_player}"
        print(state)

    def valid_moves(self):
        return self.get_proper_divisors(self.N)

    def get_proper_divisors(self, N):
        # Proper divisors are all positive integers greater than 1 and less than N that divide N exactly
        return [i for i in range(2, N) if N % i == 0]
