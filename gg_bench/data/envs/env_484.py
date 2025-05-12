import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Max starting number, used for observation_space
        self.MAX_STARTING_NUMBER = 1000

        # List of first 25 primes for action mapping
        self.primes = [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
        ]

        # Action mapping:
        # Action 0: Subtract 1
        # Actions 1 - 25: Divide by primes[action - 1]

        # Define action_space: 0 (Subtract 1) to 25 (Divide by prime 97)
        self.action_space = spaces.Discrete(26)

        # Observation space is the current number
        self.observation_space = spaces.Box(
            low=1, high=self.MAX_STARTING_NUMBER, shape=(1,), dtype=np.int32
        )

        self.current_number = None  # Current Number in the game
        self.done = False
        self.current_player = 1  # Player 1 starts

    def reset(self, seed=None, options=None, starting_number=None):
        super().reset(seed=seed)

        # Set starting number
        if starting_number is not None:
            self.current_number = starting_number
        else:
            # Set random starting number between 10 and 1000
            self.current_number = np.random.randint(10, 1001)

        self.done = False
        self.current_player = 1
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # Game is over
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        valid_actions = self.valid_moves()

        if action not in valid_actions:
            # Invalid move: current player loses
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Apply the action
        if action == 0:
            # Subtract 1
            self.current_number -= 1
        else:
            # Divide by a prime factor
            prime = self.primes[action - 1]
            self.current_number = self.current_number // prime  # Integer division

        # Check if game is over
        if self.current_number == 1:
            self.done = True
            reward = 1  # Current player wins
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Game continues
            # Switch to the next player
            self.current_player = 2 if self.current_player == 1 else 1
            reward = 0
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        return f"Current Number: {self.current_number}, Current Player: Player {self.current_player}"

    def valid_moves(self):
        valid_actions = []
        if self.current_number > 1:
            # Subtract 1 is always valid when current_number > 1
            valid_actions.append(0)
        # Check for valid division moves
        for idx, prime in enumerate(self.primes):
            if self.current_number % prime == 0:
                valid_actions.append(idx + 1)  # Action index for dividing by this prime
        return valid_actions
