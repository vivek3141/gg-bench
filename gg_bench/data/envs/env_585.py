import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Initialize the game settings
        self.initial_current_number = 100  # Starting Current Number

        # Define action and observation space
        # The action is choosing a proper divisor; indices correspond to integers from 0 to initial_current_number
        self.action_space = spaces.Discrete(self.initial_current_number + 1)

        # Observation is the Current Number
        self.observation_space = spaces.Box(
            low=1, high=self.initial_current_number, shape=(1,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize the state
        self.current_number = self.initial_current_number
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Return the initial observation and info
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Check if the action is valid
        if action < 2 or action >= self.current_number:
            # Invalid action: action must be a proper divisor
            self.done = True
            reward = -10  # Penalty for invalid move
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        if self.current_number % action != 0:
            # Invalid action: action must divide Current Number evenly
            self.done = True
            reward = -10  # Penalty for invalid move
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid action: update the Current Number
        self.current_number = self.current_number // action

        # Check if the next player has any valid moves
        valid_moves = self.get_valid_moves()
        if len(valid_moves) == 0:
            # Next player cannot make a move; current player wins
            self.done = True
            reward = 1  # Reward for winning
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Switch to the next player
            self.current_player = 2 if self.current_player == 1 else 1
            reward = 0  # No reward; game continues
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                False,
                False,
                {},
            )

    def render(self):
        # Return a string representation of the current state
        return f"Current Number: {self.current_number}, Current Player: {self.current_player}"

    def valid_moves(self):
        # Return the list of valid actions (proper divisors) at the current state
        return self.get_valid_moves()

    def get_valid_moves(self):
        # Helper function to get proper divisors of the Current Number
        valid_actions = []
        for i in range(2, self.current_number):
            if self.current_number % i == 0:
                valid_actions.append(i)
        return valid_actions
