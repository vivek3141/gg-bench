import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Game settings
        self.initial_current_number = 100  # Starting number for the game
        self.max_divisor = self.initial_current_number
        self.current_number = self.initial_current_number
        self.current_player = 1  # Player 1 starts

        # Action space: indices from 0 to max_actions
        # Actions 0 to max_actions - 1 correspond to divisors from 2 to max_divisor
        # Action max_actions corresponds to 'pass'
        self.max_actions = (
            self.max_divisor - 1
        )  # Number of potential divisors (from 2 to max_divisor)
        self.pass_action = self.max_actions  # Index for 'pass' action
        self.action_space = spaces.Discrete(
            self.max_actions + 1
        )  # Include 'pass' action

        # Observation space: Current number and current player
        # Current number ranges from 1 to initial_current_number
        # Current player is either 1 or -1
        self.observation_space = spaces.Box(
            low=np.array([1, -1]),
            high=np.array([self.initial_current_number, 1]),
            shape=(2,),
            dtype=np.int32,
        )

        self.done = False
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.initial_current_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_number, self.current_player], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.current_number, self.current_player], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Process 'pass' action
        if action == self.pass_action:
            valid_divisors = self.get_valid_divisors(self.current_number)
            if valid_divisors:
                # Passing is invalid when there are valid moves available
                self.done = True
                return (
                    np.array(
                        [self.current_number, self.current_player], dtype=np.int32
                    ),
                    -10,
                    True,
                    False,
                    {},
                )
            else:
                # Passing is valid when no valid moves are available
                # Switch player
                self.current_player *= -1
                return (
                    np.array(
                        [self.current_number, self.current_player], dtype=np.int32
                    ),
                    0,
                    False,
                    False,
                    {},
                )
        else:
            divisor = action + 2  # Map action index to actual divisor
            # Check if the divisor is valid
            if divisor <= 1 or divisor >= self.current_number:
                # Invalid divisor selected
                self.done = True
                return (
                    np.array(
                        [self.current_number, self.current_player], dtype=np.int32
                    ),
                    -10,
                    True,
                    False,
                    {},
                )
            if self.current_number % divisor != 0:
                # Divisor does not evenly divide Current Number
                self.done = True
                return (
                    np.array(
                        [self.current_number, self.current_player], dtype=np.int32
                    ),
                    -10,
                    True,
                    False,
                    {},
                )
            # Valid move; update Current Number
            self.current_number = int(self.current_number / divisor)
            # Check for win condition
            if self.current_number == 1:
                self.done = True
                return (
                    np.array(
                        [self.current_number, self.current_player], dtype=np.int32
                    ),
                    1,  # Reward for winning the game
                    True,
                    False,
                    {},
                )
            else:
                # Switch player
                self.current_player *= -1
                return (
                    np.array(
                        [self.current_number, self.current_player], dtype=np.int32
                    ),
                    0,
                    False,
                    False,
                    {},
                )

    def render(self):
        return f"Current Number: {self.current_number}, Current Player: {'1' if self.current_player == 1 else '2'}"

    def valid_moves(self):
        valid_actions = []
        valid_divisors = self.get_valid_divisors(self.current_number)
        for divisor in valid_divisors:
            action = divisor - 2  # Map divisor back to action index
            valid_actions.append(action)
        if not valid_actions:
            # Only 'pass' action is valid when no valid divisors are available
            valid_actions.append(self.pass_action)
        return valid_actions

    def get_valid_divisors(self, number):
        # Helper method to compute valid divisors of the Current Number
        valid_divisors = []
        for i in range(2, number):
            if number % i == 0:
                valid_divisors.append(i)
        return valid_divisors
