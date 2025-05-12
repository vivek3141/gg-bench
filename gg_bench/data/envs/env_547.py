import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: Actions correspond to possible divisors from 2 to 60
        # Action indices 0 to 58 correspond to divisors 2 to 60
        self.action_space = spaces.Discrete(59)

        # Define observation space: Observation is the current number
        self.observation_space = spaces.Box(low=2, high=60, shape=(1,), dtype=np.int32)

        self.starting_number = 60
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.done = False
        self.current_player = 1  # Internal tracking of player turn
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        divisor = (
            action + 2
        )  # Map action indices to divisors (0 -> 2, 1 -> 3, ..., 58 -> 60)

        # Check if the action is a valid divisor
        if divisor >= self.current_number or self.current_number % divisor != 0:
            # Invalid move: current player loses
            reward = -10
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move: subtract the divisor from the current number
        self.current_number -= divisor

        # Check if the next player has any valid moves
        next_valid_divisors = [
            d for d in range(2, self.current_number) if self.current_number % d == 0
        ]
        if not next_valid_divisors:
            # Next player has no valid moves: current player wins
            reward = 1
            self.done = True
        else:
            # Game continues: switch player
            reward = -10
            self.current_player *= -1

        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        return f"Current Number: {self.current_number}"

    def valid_moves(self):
        # Return a list of action indices corresponding to valid moves
        valid_divisors = [
            d for d in range(2, self.current_number) if self.current_number % d == 0
        ]
        return [d - 2 for d in valid_divisors]  # Map divisors to action indices
