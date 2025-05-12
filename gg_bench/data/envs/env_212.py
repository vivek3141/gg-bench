import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Game settings
        self.starting_number = 60  # Starting number for the game

        # Define action and observation space
        # The action is the divisor to subtract, ranging from 0 to starting_number
        self.action_space = spaces.Discrete(self.starting_number + 1)

        # The observation is the current number, represented as an array of one element
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([self.starting_number]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            # Game is over
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Validate the action
        if not self.is_valid_action(action):
            # Invalid move
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Apply the action
        self.current_number -= action

        # Check if the opponent can make a move
        if not self.has_valid_moves():
            # Opponent cannot move; current player wins
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player = 1 if self.current_player == 2 else 2

        return (
            np.array([self.current_number], dtype=np.int32),
            0,
            False,
            False,
            {},
        )

    def render(self):
        proper_divisors = self.get_proper_divisors(self.current_number)
        render_str = f"Current Number: {self.current_number}\n"
        render_str += f"Current Player: {self.current_player}\n"
        render_str += f"Proper Divisors: {proper_divisors}\n"
        return render_str

    def valid_moves(self):
        # Returns a list of valid actions (proper divisors of the current number)
        return self.get_proper_divisors(self.current_number)

    def is_valid_action(self, action):
        # Checks if the action is a valid proper divisor
        return action in self.get_proper_divisors(self.current_number)

    def has_valid_moves(self):
        # Checks if there are valid moves available for the next player
        return len(self.get_proper_divisors(self.current_number)) > 0

    def get_proper_divisors(self, number):
        # Returns a list of proper divisors of the given number
        divisors = []
        for i in range(2, number):
            if number % i == 0:
                divisors.append(i)
        return divisors
