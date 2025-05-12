import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=50):
        super(CustomEnv, self).__init__()

        self.starting_number = starting_number
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Action space: 0 to 8, corresponds to chosen_number = action + 1
        self.action_space = spaces.Discrete(9)

        # Observation space: the current number, an integer between 0 and starting_number
        self.observation_space = spaces.Box(
            low=0, high=self.starting_number, shape=(1,), dtype=np.int32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            # Game is over
            return np.array([self.current_number], dtype=np.int32), 0, True, False, {}

        # Check if there are any valid moves
        if not self.valid_moves():
            # Current player loses (cannot make a valid move)
            reward = -1
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        chosen_number = action + 1  # action 0 corresponds to number 1

        # Check if the action is valid
        if (chosen_number < 1 or chosen_number > 9) or (
            self.current_number % chosen_number != 0
        ):
            # Invalid move
            reward = -10
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )

        # Valid move
        self.current_number -= chosen_number

        if self.current_number == 0:
            # Current player wins
            reward = 1
            self.done = True
        else:
            # Continue the game
            # Switch to next player
            self.current_player *= -1
            reward = 0

        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        valid_divisors = [i + 1 for i in range(9) if self.current_number % (i + 1) == 0]
        return (
            f"Player {self.current_player}'s turn.\n"
            f"Current Number: {self.current_number}\n"
            f"Valid Divisors (1-9): {valid_divisors}"
        )

    def valid_moves(self):
        return [i for i in range(9) if self.current_number % (i + 1) == 0]
