import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=30, max_starting_number=1000):
        super(CustomEnv, self).__init__()

        self.starting_number = starting_number
        self.max_starting_number = max_starting_number

        # Define action space: possible actions from 0 to max_starting_number
        # These will be treated as integers representing possible factors
        self.action_space = spaces.Discrete(self.max_starting_number + 1)

        # Define observation space: the shared number, from 1 to max_starting_number
        self.observation_space = spaces.Box(
            low=np.array([1]),
            high=np.array([self.max_starting_number]),
            shape=(1,),
            dtype=np.int32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.shared_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.shared_number], dtype=np.int32), {}  # observation, info

    def step(self, action):
        # Check if game is already done
        if self.done:
            return np.array([self.shared_number], dtype=np.int32), 0, True, False, {}

        # First, check if the current player has any valid moves
        valid_moves = self.valid_moves()
        if not valid_moves:
            # No valid moves, current player loses
            self.done = True
            reward = -10  # Penalty for losing
            return (
                np.array([self.shared_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Now, check if the action is a valid move
        if action not in valid_moves:
            # Invalid move, current player loses
            self.done = True
            reward = -10  # Penalty for invalid move
            return (
                np.array([self.shared_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        # Update the shared number
        self.shared_number = self.shared_number // action

        # Check if the shared number is 1
        if self.shared_number == 1:
            # Current player wins
            self.done = True
            reward = 1  # Reward for winning
            return (
                np.array([self.shared_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Check if the next player has any valid moves
            next_valid_moves = self.valid_moves()
            if not next_valid_moves:
                # The next player cannot make any valid moves, current player wins
                self.done = True
                reward = 1  # Current player wins
                return (
                    np.array([self.shared_number], dtype=np.int32),
                    reward,
                    True,
                    False,
                    {},
                )
            else:
                # Switch to next player
                self.current_player = (
                    3 - self.current_player
                )  # Switch between player 1 and 2

                # Game continues
                reward = -1  # Penalty per move to encourage shorter games
                return (
                    np.array([self.shared_number], dtype=np.int32),
                    reward,
                    False,
                    False,
                    {},
                )

    def render(self):
        return f"Current Shared Number: {self.shared_number}"

    def valid_moves(self):
        # Returns a list of valid move actions (integers) for the current shared number
        factors = []
        n = self.shared_number
        for i in range(2, n):
            if n % i == 0:
                factors.append(i)
        return factors
