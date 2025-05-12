import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self, starting_number=60):
        super(CustomEnv, self).__init__()

        self.starting_number = starting_number
        self.current_number = None
        self.current_player = None
        self.done = None

        # Maximum possible action corresponds to starting_number - 1
        self.max_number = self.starting_number

        # Define action space: actions correspond to divisors from 2 upwards
        # So action indices range from 0 to max_number - 2
        self.action_space = spaces.Discrete(
            self.max_number - 1
        )  # actions from 2 upwards

        # Define observation space: current number (1 to starting_number)
        self.observation_space = spaces.Box(
            low=1, high=self.max_number, shape=(1,), dtype=np.int64
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = self.starting_number
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_number], dtype=np.int64), {}  # observation, info

    def step(self, action):
        if self.done:
            # Game is over
            return (
                np.array([self.current_number], dtype=np.int64),
                0,
                True,
                False,
                {},
            )  # observation, reward, terminated, truncated, info

        # Adjust action to match the proper divisor (divisors start from 2)
        selected_divisor = action + 2

        # Check if current player has any valid moves
        valid_actions = self.valid_moves()
        if len(valid_actions) == 0:
            # Current player cannot make a move, loses the game
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int64),
                reward,
                self.done,
                False,
                {},
            )

        # Validate the action
        if action not in valid_actions:
            # Invalid action, player loses the game
            self.done = True
            reward = -10
            return (
                np.array([self.current_number], dtype=np.int64),
                reward,
                self.done,
                False,
                {},
            )

        # Valid move, update the current number
        self.current_number = self.current_number // selected_divisor

        # Check for win condition
        if self.current_number == 1:
            # Current player wins by reducing number to 1
            self.done = True
            reward = 1
            return (
                np.array([self.current_number], dtype=np.int64),
                reward,
                self.done,
                False,
                {},
            )

        # Check if next player has any valid moves
        next_valid_actions = self.valid_moves()
        if len(next_valid_actions) == 0:
            # Next player cannot move, current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_number], dtype=np.int64),
                reward,
                self.done,
                False,
                {},
            )

        # Game continues, switch player
        self.current_player *= -1
        reward = 0
        return (
            np.array([self.current_number], dtype=np.int64),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        return f"Current number: {self.current_number}"

    def valid_moves(self):
        # Get proper divisors of current number
        divisors = self.get_proper_divisors(self.current_number)
        # Map divisors to action indices
        actions = [divisor - 2 for divisor in divisors]
        return actions

    @staticmethod
    def get_proper_divisors(number):
        # Proper divisors are integers greater than 1 and less than the number that divide it evenly
        return [d for d in range(2, number) if number % d == 0]
