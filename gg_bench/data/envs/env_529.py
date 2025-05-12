import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Actions correspond to digits 1-9 (actions 0-8)
        self.action_space = spaces.Discrete(9)
        # Observation is the current number (0 to 9999)
        self.observation_space = spaces.Box(
            low=0, high=9999, shape=(1,), dtype=np.int32
        )

        # Initialize the game state
        self.current_number = None
        self.current_player = None
        self.done = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_number = 0
        self.current_player = 1  # Player 1 starts
        self.done = False
        return np.array([self.current_number], dtype=np.int32), {}

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
        if not self.action_space.contains(action):
            # Invalid action
            self.done = True
            return (
                np.array([self.current_number], dtype=np.int32),
                -10,
                True,
                False,
                {},
            )
        # Convert action to digit (actions 0-8 correspond to digits 1-9)
        digit = action + 1
        # Append digit to the right of the current number
        new_number = self.current_number * 10 + digit

        if new_number > 1000:
            # Current player loses
            self.done = True
            reward = -10
        elif new_number == 1000:
            # Current player wins
            self.current_number = new_number
            self.done = True
            reward = 1
        else:
            # Game continues
            self.current_number = new_number
            self.current_player *= -1  # Switch player
            reward = 0

        return (
            np.array([self.current_number], dtype=np.int32),
            reward,
            self.done,
            False,
            {},
        )

    def render(self):
        if self.current_number == 0:
            return "Current number is empty."
        else:
            return f"Current number is: {self.current_number}"

    def valid_moves(self):
        # Returns a list of valid actions (digits that won't cause immediate loss)
        valid_actions = []
        for action in range(9):
            digit = action + 1
            new_number = self.current_number * 10 + digit
            if new_number <= 1000:
                valid_actions.append(action)
        if not valid_actions:
            # If no valid moves, all moves will result in loss
            valid_actions = list(range(9))
        return valid_actions
