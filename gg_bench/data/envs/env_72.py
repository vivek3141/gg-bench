import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define maximum N
        self.max_N = 100  # Maximum value for N
        self.min_N = 20  # Minimum starting value for N

        # Define action and observation space
        # The agent can choose any number from 0 to max_N
        self.action_space = spaces.Discrete(self.max_N + 1)
        # Observation space includes [N, current_player]
        self.observation_space = spaces.Box(
            low=np.array([2, -1]),
            high=np.array([self.max_N, 1]),
            shape=(2,),
            dtype=np.int32,
        )

        # Initialize variables
        self.N = None
        self.current_player = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize N to a random number between min_N and max_N
        self.N = self.np_random.integers(self.min_N, self.max_N + 1)
        self.current_player = 1  # Player 1 starts
        self.done = False
        observation = np.array([self.N, self.current_player], dtype=np.int32)
        return observation, {}

    def step(self, action):
        info = {}

        # Check if the game is already over
        if self.done:
            return (
                np.array([self.N, self.current_player], dtype=np.int32),
                -10,
                True,
                False,
                info,
            )

        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            # Invalid move
            reward = -10
            self.done = True
            info["invalid_move"] = True
            observation = np.array([self.N, self.current_player], dtype=np.int32)
            return observation, reward, self.done, False, info

        # Subtract the action from N
        self.N -= action

        # Check if N is prime
        if self.is_prime(self.N):
            # Current player loses
            reward = 1  # Current player wins
            self.done = True
            observation = np.array([self.N, self.current_player], dtype=np.int32)
            return observation, reward, self.done, False, info

        # Check if N is less than 2 (invalid state)
        if self.N < 2:
            # Invalid state
            reward = -10
            self.done = True
            info["invalid_state"] = True
            observation = np.array([self.N, self.current_player], dtype=np.int32)
            return observation, reward, self.done, False, info

        # Valid move, switch player
        reward = -10  # Penalize per valid move
        self.current_player *= -1  # Switch player
        observation = np.array([self.N, self.current_player], dtype=np.int32)
        return observation, reward, self.done, False, info

    def render(self):
        board_str = f"Current N: {self.N}\n"
        board_str += f"Current Player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
        board_str += f"Valid Moves: {self.valid_moves()}\n"
        return board_str

    def valid_moves(self):
        # Return a list of proper divisors of N
        divisors = []
        for i in range(2, self.N):  # Proper divisors exclude 1 and N itself
            if self.N % i == 0:
                divisors.append(i)
        return divisors

    def is_prime(self, n):
        if n <= 1:
            return False
        if n <= 3:
            return True

        if n % 2 == 0 or n % 3 == 0:
            return False

        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
