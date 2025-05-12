import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space: Multipliers from 2 to 9 (actions 0 to 7 correspond to multipliers 2 to 9)
        self.action_space = spaces.Discrete(8)

        # Define observation space: The current number in the game
        self.observation_space = spaces.Box(
            low=1, high=np.iinfo(np.int32).max, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.current_number = None
        self.used_numbers = None
        self.done = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 1
        self.used_numbers = [1]
        self.done = False
        return (
            np.array([self.current_number], dtype=np.int32),
            {},
        )  # Observation and info

    def step(self, action):
        if self.done:
            # Game has already ended
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                True,
                False,
                {},
            )

        # Map action to multiplier
        if action < 0 or action > 7:
            # Invalid action index
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        multiplier = action + 2  # Actions 0-7 correspond to multipliers 2-9

        # Calculate new number
        new_number = self.current_number * multiplier

        # Check if the new number has already been used
        if new_number in self.used_numbers:
            # Invalid move
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )

        # Valid move, update game state
        self.current_number = new_number
        self.used_numbers.append(new_number)

        # Check if the next player has any valid moves
        opponent_has_moves = False
        for m in range(2, 10):
            potential_number = self.current_number * m
            if potential_number not in self.used_numbers:
                opponent_has_moves = True
                break

        if not opponent_has_moves:
            # Current player wins
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                1,
                True,
                False,
                {},
            )
        else:
            # Game continues
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                False,
                False,
                {},
            )

    def render(self):
        state_str = (
            f"Current number: {self.current_number}\nUsed numbers: {self.used_numbers}"
        )
        return state_str

    def valid_moves(self):
        valid_actions = []
        for action in range(8):  # Actions 0-7 correspond to multipliers 2-9
            multiplier = action + 2
            new_number = self.current_number * multiplier
            if new_number not in self.used_numbers:
                valid_actions.append(action)
        return valid_actions
