import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, initial_total=100):
        super(CustomEnv, self).__init__()

        self.initial_total = initial_total

        # Define action and observation space
        # The action space represents possible integers from 0 to initial_total - 1
        # Although only positive integers less than the current total are valid at each step
        self.action_space = spaces.Discrete(self.initial_total)

        # Observation is the current total as a single integer
        self.observation_space = spaces.Box(
            low=0, high=self.initial_total, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_total = self.initial_total
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_total], dtype=np.int32), {}  # observation, info

    def step(self, action):
        if self.done:
            # If the game is over, return current state
            return np.array([self.current_total], dtype=np.int32), 0, True, False, {}

        action_number = action  # Action indices correspond to action numbers

        # Check if the action is valid
        if (
            action_number <= 0  # Must be positive
            or action_number >= self.current_total  # Must be less than current total
            or self.current_total % action_number
            != 0  # Must divide current total evenly
        ):
            # Invalid move
            self.done = True
            reward = -10  # Penalty for invalid move
            return (
                np.array([self.current_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move: subtract the action from the current total
        self.current_total -= action_number

        # Check for win condition
        if self.current_total == 0:
            # Current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check if the next player has any valid moves
        next_valid_moves = [
            i for i in range(1, self.current_total) if self.current_total % i == 0
        ]
        if not next_valid_moves:
            # Next player has no valid moves; current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_total], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player = 3 - self.current_player  # Toggles between 1 and 2
        reward = 0  # No reward for a regular valid move
        return np.array([self.current_total], dtype=np.int32), reward, False, False, {}

    def render(self):
        # Return a string representation of the current game state
        return f"Current total: {self.current_total}, Current player: {self.current_player}"

    def valid_moves(self):
        # Return a list of valid action indices (integers from 0 to initial_total - 1)
        valid_actions = [
            i for i in range(1, self.current_total) if self.current_total % i == 0
        ]
        return valid_actions
