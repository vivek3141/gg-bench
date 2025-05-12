import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Default starting Shared Number
        self.starting_shared_number = 30

        # Define action and observation space
        # Actions are integers from 0 to 100
        self.action_space = spaces.Discrete(101)

        # Observation is the current Shared Number
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([100]), shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.current_player = 1
        self.shared_number = self.starting_shared_number
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Allow setting a custom starting Shared Number via options
        if options is not None and "starting_shared_number" in options:
            self.starting_shared_number = options["starting_shared_number"]

        self.shared_number = self.starting_shared_number
        self.current_player = 1
        self.done = False

        observation = np.array([self.shared_number], dtype=np.int32)
        return observation, {}  # Return observation and info

    def get_proper_divisors(self, N):
        proper_divisors = set()
        for i in range(2, int(N**0.5) + 1):
            if N % i == 0:
                proper_divisors.add(i)
                if i != N // i:
                    proper_divisors.add(N // i)
        proper_divisors = [d for d in proper_divisors if d != N]
        return sorted(proper_divisors)

    def valid_moves(self):
        return self.get_proper_divisors(self.shared_number)

    def step(self, action):
        if self.done:
            return np.array([self.shared_number], dtype=np.int32), 0, True, False, {}

        valid_moves = self.valid_moves()

        if action not in valid_moves:
            # Invalid move
            self.done = True
            reward = -10
            return (
                np.array([self.shared_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Apply action
        self.shared_number = self.shared_number // action

        if self.shared_number == 1:
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.shared_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check if next player can make a move
        next_valid_moves = self.get_proper_divisors(self.shared_number)
        if not next_valid_moves:
            # Next player cannot move, current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.shared_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player *= -1
        reward = 0
        return (
            np.array([self.shared_number], dtype=np.int32),
            reward,
            False,
            False,
            {},
        )

    def render(self):
        return f"Current Shared Number: {self.shared_number}"

    def close(self):
        pass
