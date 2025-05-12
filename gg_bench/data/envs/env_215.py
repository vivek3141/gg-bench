import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define the starting number range
        self.min_starting_number = 10
        self.max_starting_number = 50

        # Define action and observation space

        # The action space represents possible numbers to subtract
        # Actions are integers from 0 to max_starting_number inclusive
        self.action_space = spaces.Discrete(self.max_starting_number + 1)

        # Observation is the current number
        # It's a single integer between 1 and max_starting_number
        self.observation_space = spaces.Box(
            low=np.array([1]), high=np.array([self.max_starting_number]), dtype=np.int32
        )

        # Initialize the game state
        self.reset()

    def reset(self, seed=None, options=None):
        # Initialize the random number generator
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.current_player = 1  # Player 1 starts
        self.done = False

        # Randomly select starting number
        self.starting_number = self.np_random.integers(
            self.min_starting_number, self.max_starting_number + 1
        )
        self.current_number = self.starting_number

        return np.array([self.current_number], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            return (
                np.array([self.current_number], dtype=np.int32),
                0,
                True,
                False,
                {"error": "Game is already over"},
            )
        # Check if action is valid
        valid_actions = self.valid_moves()
        if action not in valid_actions:
            self.done = True
            reward = -10  # Invalid move
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {"error": "Invalid action"},
            )
        # Valid action
        self.current_number -= action

        # Check for win condition
        if len(self.get_proper_divisors(self.current_number)) == 0:
            # Opponent cannot make a move, current player wins
            self.done = True
            reward = 1
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                True,
                False,
                {},
            )
        else:
            # Switch player
            self.current_player *= -1
            reward = 0
            return (
                np.array([self.current_number], dtype=np.int32),
                reward,
                False,
                False,
                {},
            )

    def valid_moves(self):
        # Return a list of valid actions (proper divisors excluding 1 and the number itself)
        return self.get_proper_divisors(self.current_number)

    def get_proper_divisors(self, number):
        # Helper method to get proper divisors
        divisors = []
        for i in range(2, number):  # Exclude 1 and the number itself
            if number % i == 0:
                divisors.append(i)
        return divisors

    def render(self):
        # Return a string showing the current state
        return f"Current number: {self.current_number}\nValid moves: {self.valid_moves()}\nCurrent player: {'Player 1' if self.current_player == 1 else 'Player 2'}\n"
