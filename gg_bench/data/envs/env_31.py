import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the starting number
        self.starting_number = 60

        # Define action and observation space
        # The action space is Discrete with possible actions from 0 to starting_number
        self.action_space = spaces.Discrete(self.starting_number + 1)

        # The observation space is a Box space representing the current number
        self.observation_space = spaces.Box(
            low=1, high=self.starting_number, shape=(1,), dtype=np.int32
        )

        # Initialize the environment
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
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        # Get valid moves
        valid_actions = self.valid_moves()

        # Check if the action is valid
        if action not in valid_actions:
            # Invalid move
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
        self.current_number = self.current_number // action

        # Check for winning condition
        if self.current_number == 1:
            # Current player wins by reducing the number to 1
            self.done = True
            reward = 1
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Check if the opponent has any valid moves
        opponent_valid_moves = self.valid_moves()
        if len(opponent_valid_moves) == 0:
            # Opponent cannot make a move; current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Switch to the next player
        self.current_player = (
            3 - self.current_player
        )  # Switch between Player 1 (1) and Player 2 (2)
        reward = 0
        return np.array([self.current_number], dtype=np.int32), reward, False, False, {}

    def render(self):
        # Return a string representation of the current game state
        s = f"Current Number: {self.current_number}\n"
        s += f"Current Player: Player {self.current_player}\n"
        valid_actions = self.valid_moves()
        if valid_actions:
            s += f"Proper Divisors: {', '.join(map(str, valid_actions))}\n"
        else:
            s += "Proper Divisors: None (number is prime)\n"
        return s

    def valid_moves(self):
        # Return a list of valid proper divisors for the current number
        return [
            d for d in range(2, self.current_number) if self.current_number % d == 0
        ]
