import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=100):
        super(CustomEnv, self).__init__()

        self.starting_number = starting_number
        self.max_divisor = starting_number

        # Define action space: Divisors from 2 up to max_divisor inclusive
        # Action indices from 0 to max_divisor - 2
        self.action_space = spaces.Discrete(self.max_divisor - 1)

        # Observation space: Current number in the game
        self.observation_space = spaces.Box(
            low=1, high=self.starting_number, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        divisor = action + 2  # Map action index to divisor

        # Validate action
        if (
            self.current_number % divisor != 0
            or divisor <= 1
            or divisor > self.max_divisor
        ):
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Apply move
        self.current_number = self.current_number // divisor

        # Check for win condition
        if self.current_number == 1:
            self.done = True
            return np.array([self.current_number], dtype=np.int32), 1, True, False, {}

        # Check if next player has any valid moves
        next_valid_moves = self.valid_moves()
        if not next_valid_moves:
            self.done = True
            return np.array([self.current_number], dtype=np.int32), 1, True, False, {}

        # Switch player
        self.current_player = 3 - self.current_player

        return np.array([self.current_number], dtype=np.int32), 0, False, False, {}

    def render(self):
        return (
            f"Player {self.current_player}'s Turn:\n"
            f"Current Number: {self.current_number}\n"
            f"Available Divisors: {', '.join(str(d) for d in self.get_available_divisors())}\n"
        )

    def valid_moves(self):
        valid_actions = []
        for divisor in range(2, self.max_divisor + 1):
            if self.current_number % divisor == 0:
                action_index = divisor - 2
                valid_actions.append(action_index)
        return valid_actions

    def get_available_divisors(self):
        # Helper function to get the list of available divisors
        return [
            divisor
            for divisor in range(2, self.max_divisor + 1)
            if self.current_number % divisor == 0
        ]
